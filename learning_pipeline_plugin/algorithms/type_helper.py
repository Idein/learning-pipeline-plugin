import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar
from typing_extensions import TypedDict  # type: ignore

import numpy as np
from PIL.Image import Image


class _DataDictMendatory(TypedDict):
    """Class for passing data
    `image` must be PIL.Image
    `feature_vector` must be of shape (N,)
    """
    image: Image
    feature_vector: np.ndarray


class DataDict(_DataDictMendatory, total=False):
    other_data: Dict[str, Any]


class DataSample:
    def __init__(self, data: DataDict):
        self.image = data["image"]
        self.feature_vector = data["feature_vector"]
        self.timestamp = datetime.now(timezone.utc).isoformat()  # UTC, isoformat
        self.other_data = data.get("other_data", {})
