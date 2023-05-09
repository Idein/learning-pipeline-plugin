import io
import json
import os
import random
import time
from typing import Dict, Generic, Optional, Tuple, TypeVar, Union

import requests
from actfw_core.service_client import ServiceClient
from PIL.Image import Image

from .actfw_utils import IsolatedTask
from .notifier import AbstractNotifier, NullNotifier

T = TypeVar("T")
UserMetadata = Dict[str, Union[str, int, float, bool]]
DatedImage = Tuple[str, Image]


class AbstractSenderTask(Generic[T], IsolatedTask[T]):
    def set_notifier(self, notifier: AbstractNotifier) -> None:
        raise NotImplementedError

    def _proc(self, data: T) -> None:
        raise NotImplementedError


class SenderTask(AbstractSenderTask[DatedImage]):
    def __init__(self,
                 pipeline_id: str,
                 metadata: Optional[UserMetadata] = None,
                 endpoint_root: str = "https://api.autolearner.actcast.io",
                 max_retries: int = 3,
                 backoff_time_base: float = 60.0,
                 inqueuesize: int = 0):
        """Isolated task used to send data to the Learning pipeline servers.
        - pipeline_id (str): ID of the pipeline to send data to (obtained after created a pipeline)
        - metadata (UserMetadata): JSON-like data that will be stored with the image
                                    (e.g. user may include here some act settings)
        - endpoint_root (str): endpoint root of the lp API server (https://....)
        - max_retries (int): Max number of retry attempts.
        - backoff_time_base (float): The delay time between retry attempts
          is defined by random(backoff_time_base * (2 ** i)) seconds.
        - inqueuesize (int): size of the sending queue (default: 0 (no limit))

        Use example:
        ```
        st = SenderTask(pipeline_id)
        app.register_task(st)
        st.set_notifier(my_notifier)

        ...
        st.enqueue((time_stamp, image))
        ```
        """
        super().__init__(inqueuesize)
        self.service_client = ServiceClient()
        self.endpoint_root = endpoint_root
        self.pipeline_id = pipeline_id
        self.notifier: AbstractNotifier = NullNotifier()
        self.user_metadata = {} if metadata is None else metadata
        self.device_token = None
        self.device_token_endpoint = os.path.join(endpoint_root, "device", "token")
        self.collect_requests_endpoint = os.path.join(endpoint_root, "collect", "requests")
        assert max_retries > 0
        self.max_retries = max_retries
        assert backoff_time_base > 0
        self.backoff_time_base = backoff_time_base

    def set_notifier(self, notifier: AbstractNotifier) -> None:
        self.notifier = notifier

    def backoff(self, attempt: int) -> None:
        assert attempt >= 0
        max_period = self.backoff_time_base * (2 ** attempt)
        t = random.random() * max_period
        time.sleep(t)

    def proxies_common(self) -> dict:
        return {"https": f'socks5h://{os.environ["ACTCAST_SOCKS_SERVER"]}'}

    def has_valid_params(self) -> bool:
        if self.endpoint_root == "":
            self.notifier.notify("The specified endpoint URL is blank.")
            return False

        if os.environ.get("ACTCAST_GROUP_ID") is None:
            self.notifier.notify("Failed to identify Group ID.")
            return False

        return True

    def has_valid_token(self) -> bool:
        return self.device_token is not None and self.device_token_expires > time.time()

    def update_token(self) -> None:
        def generate_headers(sending_context: dict) -> dict:
            def generate_signature(object: dict) -> str:
                return self.service_client.rs256(json.dumps(object, sort_keys=True).encode("ascii"))

            signature = generate_signature(sending_context)
            headers: Dict[str, str] = {}
            headers.update(**sending_context, **{"Authorization": signature})

            return headers

        sending_context = {
            "device_id": os.environ["ACTCAST_DEVICE_ID"],
            "group_id": os.environ["ACTCAST_GROUP_ID"],
            "pipeline_id": self.pipeline_id
        }

        response = requests.get(
            self.device_token_endpoint,
            headers=generate_headers(sending_context),
            proxies=self.proxies_common()
        )

        if response.status_code == 200:
            try:
                payload = response.json()
                device_token = payload["data_collect_token"]
                expires_in = payload["expires_in"]
            except (requests.exceptions.JSONDecodeError, KeyError, TypeError):
                self.device_token = None
                self.notifier.notify(f"Invalid API response. (endpoint: {self.device_token_endpoint})")
                return

            self.device_token = device_token
            self.device_token_expires = time.time() + expires_in
        else:
            self.device_token = None
            failure_status = {"status_code": response.status_code, "text": response.text}
            self.notifier.notify(f"Failed to update Device Token. {failure_status}")

    def send_image(self, data: DatedImage) -> bool:
        def get_upload_url(timestamp: str) -> Optional[str]:
            response = requests.post(
                self.collect_requests_endpoint,
                json={
                    "timestamp": timestamp,
                    "act_id": os.environ.get("ACTCAST_ACT_ID"),
                    "user_data": json.dumps(self.user_metadata)
                },
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer {}".format(self.device_token),
                },
                proxies=self.proxies_common()
            )
            if response.status_code == 200:
                try:
                    payload = response.json()
                    upload_url = payload["url"]
                    return upload_url
                except (requests.exceptions.JSONDecodeError, KeyError, TypeError):
                    self.notifier.notify(f"Invalid API response. (endpoint: {self.collect_requests_endpoint})")
                    return None
            else:
                failure_status = {"status_code": response.status_code, "text": response.text}
                self.notifier.notify(f"Failed to get an upload url. {failure_status}")
                return None

        def upload_image(upload_url: str, image: Image) -> bool:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")

            response = requests.put(upload_url, data=image_bytes.getvalue(), proxies=self.proxies_common())

            if response.status_code == 200:
                return True
            else:
                failure_status = {"status_code": response.status_code, "text": response.text}
                self.notifier.notify(f"Failed to upload an image to {upload_url} . {failure_status}")
                return False

        timestamp, image = data
        upload_url = get_upload_url(timestamp)

        if upload_url is None:
            return False

        return upload_image(upload_url, image)

    def _proc(self, data: DatedImage) -> None:
        if not self.has_valid_params():
            return

        success = False
        for i in range(self.max_retries):
            if not self.has_valid_token():
                self.update_token()

                if not self.has_valid_token():
                    self.backoff(i)
                    continue

            success = self.send_image(data)

            if success:
                break

            self.backoff(i)

        result = "Success" if success else "Failure"
        self.notifier.notify(f"SenderTask Result: {result}")
