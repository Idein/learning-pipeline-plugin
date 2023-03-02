import base64
import io
import json
import os
import time
from typing import Dict, Generic, Tuple, TypeVar, Union

import requests
from actfw_core.service_client import ServiceClient
from PIL.Image import Image

from .actfw_utils import IsolatedTask
from .notifier import AbstractNotifier, NullNotifier

T = TypeVar("T")
UserMetadata = Dict[str, Union[str, int, float, bool]]
DatedImage = Tuple[str, Image]


class SendingError(Exception):
    pass


class AbstractSenderTask(Generic[T], IsolatedTask[T]):
    def set_notifier(self, notifier: AbstractNotifier) -> None:
        raise NotImplementedError

    def _proc(self, data: T) -> None:
        raise NotImplementedError


class SenderTask(AbstractSenderTask[DatedImage]):
    def __init__(self,
                 pipeline_id: str,
                 metadata: UserMetadata = {},
                 endpoint_root: str = "https://api.autolearner.actcast.io",
                 inqueuesize: int = 0):
        """Isolated task used to send data to the Learning pipeline servers.
        - pipeline_id (str): ID of the pipeline to send data to (obtained after created a pipeline)
        - metadata(UserMetadata): JSON-like data that will be stored with the image
                                    (e.g. user may include here some act settings)
        - endpoint_root(str): endpoint root of the lp API server (https://....)
        - inqueuesize(int): size of the sending queue (default: 0 (no limit))

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
        # notifier to be set through `set_notifier()` method called by CollectPipe
        self.notifier: AbstractNotifier = NullNotifier()
        self.user_metadata = json.dumps(metadata)
        self.data_collect_token = None
        self._sending_enabled = True

    def set_notifier(self, notifier: AbstractNotifier) -> None:
        """Set the notifier used by collect_pipe
        - notifier(AbstractNotifier): message formatter to notify sending success/failure to Actcast
        """
        self.notifier = notifier

    @property
    def data_collect_url(self) -> str:
        return os.path.join(self.endpoint_root, "data_collect")

    @property
    def request_data_collect_token_url(self) -> str:
        return os.path.join(self.endpoint_root, "device", "token")

    def is_sending_capable(self) -> bool:
        if self._sending_enabled:
            if self.endpoint_root == "":
                self.notifier.notify("endpoint URL is not set, data sending will fail")
                self._sending_enabled = False
            if os.environ.get("ACTCAST_GROUP_ID") is None:
                self.notifier.notify("Group ID could not be retrieved, check device firmware")
                self._sending_enabled = False
        return self._sending_enabled

    def _proc(self, data: DatedImage) -> None:
        timestamp, image = data
        pngimage = io.BytesIO()
        image.save(pngimage, format="PNG")
        b64_image = base64.b64encode(pngimage.getbuffer()).decode("utf-8")

        success = self._retry_call_api(timestamp, b64_image, max_retry=3)
        self.notifier.notify("Tried to send data sample: {}".format(
            "Success" if success else "Failure"))

    def _retry_call_api(self,
                        timestamp: str,
                        b64_image: str,
                        max_retry: int,
                        retry_count: int = 0) -> bool:
        """recursive method to try multiple times to send data
        returns sending success status
        """
        if retry_count == max_retry:
            return False

        try:
            status_code, text = self._call_api(timestamp, b64_image)
            if status_code == 200:
                return True
            elif status_code == 401:
                self._request_data_collect_token()
        except SendingError:
            self.notifier.notify("Data Collect Token request failure")
            return False
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError):
            return self._retry_call_api(timestamp, b64_image, max_retry, retry_count+1)
        else:
            self.notifier.notify(f"Data collect failed with status {status_code} (reason: {text})")
            return self._retry_call_api(timestamp, b64_image, max_retry, retry_count+1)

    def _call_api(self, timestamp: str, b64_image: str) -> Tuple[int, str]:
        """Send to the server
        returns status code of request
        """
        if not self.is_sending_capable():
            raise SendingError()
        if self.data_collect_token is None or self.data_collect_token_expires < time.time():
            self._request_data_collect_token()

        resp = requests.post(
            self.data_collect_url,
            json={
                "timestamp": timestamp,
                "image": b64_image,
                "device_id": os.environ.get("ACTCAST_DEVICE_ID"),
                "act_id": os.environ.get("ACTCAST_ACT_ID"),
                "pipeline_id": self.pipeline_id,
                "user_data": self.user_metadata
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {}".format(self.data_collect_token),
            },
            proxies={"https": f'socks5h://{os.environ["ACTCAST_SOCKS_SERVER"]}'},
        )
        return resp.status_code, resp.text

    def _request_data_collect_token(self) -> None:
        """Fetch authorization token from server, which is required for data sending
        """
        if not self._sending_enabled:
            return
        data_collect_token = None
        headers = {
            "device_id": os.environ["ACTCAST_DEVICE_ID"],
            "group_id": os.environ["ACTCAST_GROUP_ID"],
            "pipeline_id": self.pipeline_id
        }

        signature = self.service_client.rs256(json.dumps(headers, sort_keys=True).encode("ascii"))

        headers["Authorization"] = signature
        resp = requests.get(
            self.request_data_collect_token_url,
            headers=headers,
            proxies={"https": f'socks5h://{os.environ["ACTCAST_SOCKS_SERVER"]}'},
        )
        if resp.status_code == 200:
            payload = resp.json()
            data_collect_token = payload["data_collect_token"]
            expires_in = payload["expires_in"]
            self.data_collect_token = data_collect_token
            self.data_collect_token_expires = time.time() + expires_in
        else:
            self.notifier.notify(f"Data Collect Token request failed with status {resp.status_code}"
                                 f" (reason: {resp.text})")
            resp.raise_for_status()
