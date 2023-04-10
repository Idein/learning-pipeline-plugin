import json
import time
from typing import Tuple

import learning_pipeline_plugin
import pytest
import responses
from actfw_core.service_client import ServiceClient
from PIL import Image
from pytest import MonkeyPatch
from requests import PreparedRequest

ENDPOINT = "https://api.mock.autolearner.actcast.io"
PUT_URL = "https://pseudo.url/test"




def device_token_callback(request: PreparedRequest) -> Tuple[int, dict, str]:
    request_headers = request.headers
    assert {"device_id", "group_id", "pipeline_id", "Authorization"}.issubset(request_headers.keys())
    resp_body = {
        "data_collect_token": "b1af81f2-5622-4ae2-9e89-4344a47337ce",
        "expires_in": 3
    }
    return (200, {}, json.dumps(resp_body))

def collect_requests_callback(request: PreparedRequest) -> Tuple[int, dict, str]:
    request_headers = request.headers
    assert {"Accept", "Authorization"}.issubset(request_headers.keys())
    assert request.body is not None
    request_body = json.loads(request.body)
    assert {"timestamp", "act_id", "user_data"}.issubset(request_body.keys())
    resp_body = {
        "url": PUT_URL
    }
    return (200, {}, json.dumps(resp_body))


def prepare(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("ACTCAST_SERVICE_SOCK", "")
    monkeypatch.setenv("ACTCAST_SOCKS_SERVER", "")
    monkeypatch.setenv("ACTCAST_DEVICE_ID", "qwe-123-rty")
    monkeypatch.setenv("ACTCAST_GROUP_ID", "123456")

    def pseudo_rs256(self, payload: bytes) -> str:
        return "pseudo_sign"
    ServiceClient.rs256 = pseudo_rs256

    # mock device token API
    responses.add_callback(
        responses.GET,
        f"{ENDPOINT}/device/token",
        callback=device_token_callback,
        content_type="application/json",
    )

    # mock collect request API
    responses.add_callback(
        responses.POST,
        f"{ENDPOINT}/collect/requests",
        callback=collect_requests_callback,
        content_type="application/json",
    )
    responses.put(
        PUT_URL
    )


@responses.activate
def test_sender_update_token(monkeypatch: MonkeyPatch):
    prepare(monkeypatch)
    sender = learning_pipeline_plugin.sender_task.SenderTask(
        "123",
        endpoint_root=ENDPOINT
    )

    # device token is not set yet
    responses.assert_call_count(f"{ENDPOINT}/device/token", 0)
    assert sender.device_token is None
    assert getattr(sender, "device_token_expires", None) is None

    sender.update_token()

    # device token and expiration are set
    responses.assert_call_count(f"{ENDPOINT}/device/token", 1)
    assert sender.device_token is not None
    assert sender.device_token_expires > time.time()


@responses.activate
def test_sender_send_image(monkeypatch: MonkeyPatch):
    prepare(monkeypatch)
    sender = learning_pipeline_plugin.sender_task.SenderTask(
        "123",
        endpoint_root=ENDPOINT
    )

    # before collect
    responses.assert_call_count(f"{ENDPOINT}/collect/requests", 0)
    responses.assert_call_count(PUT_URL, 0)

    dated_image = ("", Image.new(mode="RGB", size=(200, 200)))
    sender.send_image(dated_image)

    # one call each
    responses.assert_call_count(f"{ENDPOINT}/collect/requests", 1)
    responses.assert_call_count(PUT_URL, 1)


@responses.activate
def test_sender_e2e(monkeypatch: MonkeyPatch):
    prepare(monkeypatch)
    sender = learning_pipeline_plugin.sender_task.SenderTask(
        "123",
        endpoint_root=ENDPOINT
    )

    # before call
    responses.assert_call_count(f"{ENDPOINT}/device/token", 0)
    responses.assert_call_count(f"{ENDPOINT}/collect/requests", 0)
    responses.assert_call_count(PUT_URL, 0)

    dated_image = ("", Image.new(mode="RGB", size=(200, 200)))
    sender.start()
    sender.enqueue(dated_image)
    time.sleep(1)
    sender.enqueue(dated_image)
    time.sleep(3)
    sender.enqueue(dated_image)
    sender.stop()
    sender.join()

    """
    flow is as follow:

    ```text
    [first enqueue]
    get token
    get collect url
    put image

    [second enqueue]
    # (token is still valid)
    get collect url
    put image

    [third enqueue]
    get token  # (token got at first enqueue expired)
    get collect url
    put image
    ```

    so we expect:
    - get token: 2 calls
    - get collect url: 3 calls
    - put image: 3 calls
    """

    responses.assert_call_count(f"{ENDPOINT}/device/token", 2)
    responses.assert_call_count(f"{ENDPOINT}/collect/requests", 3)
    responses.assert_call_count(PUT_URL, 3)
