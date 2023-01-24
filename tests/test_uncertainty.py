from typing import Any, Dict

from PIL import Image
import numpy as np

import learning_pipeline_plugin
from learning_pipeline_plugin import uncertainty
from learning_pipeline_plugin.algorithms.type_helper import DataDict, DataSample


def prepare_inputs(other_data: Dict[str, Any]) -> DataSample:
    return DataSample({
        "image": Image.fromarray(np.zeros((3,3,3)).astype("uint8")),
        "feature_vector": np.zeros(5),
        "other_data": other_data
    })


def softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    z = np.sum(np.exp(x-m))
    return np.exp(x-m) / z


def test_pseudo_uncertainty():
    unc = uncertainty.PseudoUncertainty()
    value = unc(prepare_inputs({}))
    assert isinstance(value, float)


def test_classification_uncertainty():
    unc = learning_pipeline_plugin.uncertainty.ClassificationUncertainty()

    one_shot_input = prepare_inputs({"probabilities": np.array([1., 0, 0, 0, 0])})
    assert unc(one_shot_input) == 0.

    random_probs = softmax(np.abs(np.random.randn(10)))
    random_input = prepare_inputs({"probabilities": random_probs})
    assert isinstance(unc(random_input), float)
