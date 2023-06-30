import io
import json
import os
import random
import time
from typing import Dict, Generic, Optional, Tuple, Type, TypeVar, Union

import requests
from actfw_core.service_client import ServiceClient
from PIL.Image import Image
from typing_extensions import Protocol

from .actfw_utils import IsolatedTask
from .notifier import AbstractNotifier, NullNotifier

T = TypeVar("T")
UserMetadata = Dict[str, Union[str, int, float, bool]]
DatedImage = Tuple[str, Image]


class ServiceClientProtocol(Protocol):
    def rs256(self, payload: bytes) -> str:
        raise NotImplementedError


class AbstractSenderTask(Generic[T], IsolatedTask[T]):
    def set_notifier(self, notifier: AbstractNotifier) -> None:
        raise NotImplementedError

    def _proc(self, data: T) -> None:
        _ = self._sender_call(data)

    def _sender_call(self, data: T) -> bool:
        """Should return the boolean equal to the success of the operation.
        Can be used by subclass to perform operations on success or failure
        """
        raise NotImplementedError


class AbstractSenderTaskGenericDated(Generic[T], AbstractSenderTask[T]):
    ServiceClient: Type[ServiceClientProtocol] = ServiceClient

    def __init__(self,
                 pipeline_id: str,
                 metadata: Optional[UserMetadata] = None,
                 endpoint_root: str = "https://api.autolearner.actcast.io",
                 connection_timeout: float = 3.0,
                 read_timeout: float = 29.0,
                 max_retries: int = 3,
                 backoff_time_base: float = 60.0,
                 inqueuesize: int = 0):
        """Abstract isolated task used to send data to the Learning pipeline servers.
        A concrete subclass should implement the two static methods: `timestamp_from_data` and
        `bytes_from_data`.

        - pipeline_id (str): ID of the pipeline to send data to (obtained after created a pipeline)
        - metadata (UserMetadata): JSON-like data that will be stored with the image
                                    (e.g. user may include here some act settings)
        - endpoint_root (str): endpoint root of the lp API server (https://....)
        - connection_timeout (float): number of connection timeout seconds. (default: 3.0)
        - read_timeout (float): number of timeout seconds. (default: 29.0)
          this parameter is set for connect timeout and read timeout (default: 29.0)
        - max_retries (int): Max number of retry attempts.
        - backoff_time_base (float): The delay time between retry attempts
          is defined by random(backoff_time_base * (2 ** i)) seconds.
        - inqueuesize (int): size of the sending queue (default: 0 (no limit))

        Use example:
        ```
        st = MySenderTask(pipeline_id)
        app.register_task(st)
        st.set_notifier(my_notifier)

        ...
        st.enqueue(my_data)
        ```
        """
        super().__init__(inqueuesize)
        self.service_client = self.ServiceClient()
        self.endpoint_root = endpoint_root
        self.connection_timeout = connection_timeout
        self.read_timeout = read_timeout
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

    @staticmethod
    def timestamp_from_data(data: T) -> str:
        """Returns a ISO formated timestamp of the data.
        """
        raise NotImplementedError

    @staticmethod
    def bytes_from_data(data: T) -> bytes:
        """Returns a bytes object corresponding to the data to be sent.
        e.g. converts the image inside data to PNG format and returns its bytes.
        """
        raise NotImplementedError

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

        self.device_token = None
        try:
            response = requests.get(
                self.device_token_endpoint,
                headers=generate_headers(sending_context),
                proxies=self.proxies_common(),
                timeout=(self.connection_timeout, self.read_timeout)
            )
        except requests.exceptions.ConnectTimeout:
            self.notifier.notify(f"Connection timeout (endpoint: {self.device_token_endpoint})")
            return
        except requests.exceptions.ReadTimeout:
            self.notifier.notify(f"Read timeout (endpoint: {self.device_token_endpoint})")
            return
        except requests.exceptions.RequestException:
            self.notifier.notify(f"Request failure (endpoint: {self.collect_requests_endpoint})")
            return

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

    def send_image(self, data: T) -> bool:
        def get_upload_url(data: T) -> Optional[str]:
            timestamp = self.timestamp_from_data(data)
            try:
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
                    proxies=self.proxies_common(),
                    timeout=(self.connection_timeout, self.read_timeout)
                )
            except requests.exceptions.ConnectTimeout:
                self.notifier.notify(f"Connection timeout (endpoint: {self.collect_requests_endpoint})")
                return None
            except requests.exceptions.ReadTimeout:
                self.notifier.notify(f"Read timeout (endpoint: {self.collect_requests_endpoint})")
                return None
            except requests.exceptions.RequestException:
                self.notifier.notify(f"Request failure (endpoint: {self.collect_requests_endpoint})")
                return None

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

        def upload_image(upload_url: str, data: T) -> bool:
            image_bytes = self.bytes_from_data(data)

            try:
                response = requests.put(
                    upload_url,
                    data=image_bytes,
                    proxies=self.proxies_common(),
                    timeout=(self.connection_timeout, self.read_timeout)
                )
            except requests.exceptions.ConnectTimeout:
                self.notifier.notify(f"Connection timeout (endpoint: {upload_url})")
                return False
            except requests.exceptions.ReadTimeout:
                self.notifier.notify(f"Read timeout (endpoint: {upload_url})")
                return False
            except requests.exceptions.RequestException:
                self.notifier.notify(f"Request failure (endpoint: {self.collect_requests_endpoint})")
                return False

            if response.status_code == 200:
                return True
            else:
                failure_status = {"status_code": response.status_code, "text": response.text}
                self.notifier.notify(f"Failed to upload an image to {upload_url} . {failure_status}")
                return False

        upload_url = get_upload_url(data)

        if upload_url is None:
            return False

        return upload_image(upload_url, data)

    def _sender_call(self, data: T) -> bool:
        if not self.has_valid_params():
            return False

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
        return success


class SenderTask(AbstractSenderTaskGenericDated[DatedImage]):
    """Isolated task to send data in format (timestamp, image),
    where timestamp is a ISO formatted string, and image a pillow image.
    See super class docstring for more informations.

    Use example:
    ```
    st = SenderTask(pipeline_id)
    app.register_task(st)
    st.set_notifier(my_notifier)

    ...
    st.enqueue((time_stamp, image))
    """
    @staticmethod
    def bytes_from_data(data: DatedImage) -> bytes:
        _, image = data
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format="PNG")
        return image_bytes_io.getvalue()

    @staticmethod
    def timestamp_from_data(data: DatedImage) -> str:
        timestamp, _ = data
        return timestamp
