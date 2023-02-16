import time
from typing import List

from .actfw_utils import IsolatedTaskSingleBuffer
from .algorithms import submodular_utils
from .algorithms.stream_al import MemoizationSieveStreamingPlusPlusSelector
from .algorithms.submodular_diversity import SubmodularDiversity
from .algorithms.type_helper import DataDict, DataSample
from .algorithms.uncertainty import AbstractUncertainty
from .sender_task import AbstractSenderTask


def _initialize_selector(uncertainty: AbstractUncertainty,
                         batch_size: int,
                         epsilon: float,
                         unc_div_lambda: float) -> MemoizationSieveStreamingPlusPlusSelector[DataSample]:
    """Initialize SSPP selector from parameters
    """
    submodular_funcs: List[submodular_utils.AbstractSubmodular] = []
    funcs_weights: List[float] = []
    if unc_div_lambda > 0:
        # use uncertainty:
        unc = submodular_utils.SubmodularPolicy(uncertainty)
        submodular_funcs.append(unc)
        funcs_weights.append(unc_div_lambda)
    if unc_div_lambda < 1:
        # use diversity
        div = SubmodularDiversity(div_type='determinant',
                                  sim_metric='inner_product',
                                  alpha=10,
                                  preprocess_params={"post_norm": True},
                                  memoization=True,
                                  preprocess_memoization=True)
        submodular_funcs.append(div)
        funcs_weights.append(1. - unc_div_lambda)

    objective = submodular_utils.SubmodularLC(submodular_funcs, funcs_weights)
    selector = MemoizationSieveStreamingPlusPlusSelector(batch_size, f=objective, eps=epsilon)
    return selector


class SelectTask(IsolatedTaskSingleBuffer[DataDict]):

    def __init__(self,
                 uncertainty: AbstractUncertainty,
                 epsilon: float,
                 batch_size: int,
                 sender: AbstractSenderTask,
                 unc_div_lam: float = 0.91,
                 freq: int = 60):  # minute
        """SelectTask task determines the data to be collected and put the data to SenderTask.
        - uncertainty(AbstractUncertainty): function to evaluate the uncertainty of an image
        - epsilon(float): trade-off parameter between accuracy and computational complexity.
                        The smaller the parameter, the higher the accuracy, but the increased computational complexity.
        - batch_size(int): parameter for how many images to collect per send.
        - sender(AbstractSenderTask): instance of SenderTask
        - unc_div_lam(float): weighting parameter to balance uncertainty/diversity
                        (diversity only if 0, uncertainty only if 1)
        - freq(int): frequency of determining whether or not to send data

        Use example:
        ```
        select_task = SelectTask(uncertainty, 0.01, 16, sender_task)
        app.register_task(select_task)
        ...
        select_task.enqueue({'image': image, 'feature_vector': feature_vector, 'other_data': {}})
        ```
        `feature_vector` must be of shape (N,).
        """
        super().__init__()

        self._selector = _initialize_selector(
            uncertainty,
            batch_size,
            epsilon,
            unc_div_lam
        )
        self.sender = sender
        self._idx = 0
        self.freq = freq * 60  # second
        self.substream_end = time.time() + self.freq

    def _proc(self, data: DataDict) -> None:
        sample = DataSample(data)
        self._idx += 1

        self._selector.judge(sample)

        if time.time() > self.substream_end:
            for selected_sample in self._selector.get_selected_batch():
                self.sender.enqueue((
                    selected_sample.timestamp,
                    selected_sample.image
                ))
            self._selector.reset()
            self._idx = 0
            self.substream_end = time.time() + self.freq
