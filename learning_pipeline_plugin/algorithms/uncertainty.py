from typing import Generic

import numpy as np

from .submodular_utils import T
from .type_helper import DataSample

import numpy as np


class AbstractUncertainty(Generic[T]):
    def __call__(self, data: T) -> float:
        raise NotImplementedError


class PseudoUncertainty(AbstractUncertainty):
    def __call__(self, data: T) -> float:
        return 1.


class ClassificationUncertainty(AbstractUncertainty[DataSample]):
    """Uncertainty function designed for classification tasks.
    When passing a DataDict, `other_data` sub-dict must contain the following key:
    - `probabilities`: np.ndarray containing the probabilities for each label.
                This array is expected to contain values in [0., 1.], such as `np.sum(probabilities) == 1.`
    """
    def __call__(self, data: DataSample) -> float:
        p = data.other_data["probabilities"]
        return float(-np.sum(p * np.log(np.where(p > 0, p, 1))))


class DiffFromPrevUncertainty(AbstractUncertainty[DataSample]):
    """Uncertainty function designed for general task."""

    def __init__(self):
        self.prev = None

    def __call__(self, data: DataSample) -> float:
        ftr = data.feature_vector

        if self.prev is None:
            v = 0.0
        else:
            v = float(np.linalg.norm(self.prev-ftr, ord=1))

        self.prev = ftr

        return v


class DetectionUncertainty(AbstractUncertainty[DataSample]):
    """Uncertainty function designed for detection tasks.
    When passing a DataDict, `other_data` sub-dict must contain the following key:
    - `detections`: np.ndarray including the probability of each class in the bbox regions.
        This array is expected to contain values in [0., 1.] and shape is (bbox_num, class_num+1),
        such as `candidates.sum(axis=1) == np.ones(bbox_num)`
    """

    def _least_confident(self, candidates: np.ndarray) -> float:
        if len(candidates) == 0:
            uncertainty = 0.
        else:
            uncertainty = max(*map(lambda x: 1 - x[0], candidates), 0)
        return uncertainty

    def __call__(self, data: DataSample) -> float:
        candidates = data.other_data["detections"]
        uncertainty = self._least_confident(candidates)
        return uncertainty
