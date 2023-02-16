from typing import Generic, Optional, TypeVar

from actfw_core import Application
from actfw_core.task import Pipe

from .algorithms.type_helper import DataDict
from .algorithms.uncertainty import AbstractUncertainty
from .notifier import AbstractNotifier, Notifier
from .select_task import SelectTask
from .sender_task import AbstractSenderTask

D = TypeVar("D")


class CollectPipeBase(Generic[D], Pipe[D, D]):
    def __init__(
        self,
        app: Application,
        sender_task: AbstractSenderTask,
        uncertainty: AbstractUncertainty,
        collect_frequency: int,
        batch_size: int,
        sspp_eps: float = 0.05,
        uncertainty_rate_lambda: float = 0.91,
        notifier: AbstractNotifier = Notifier(),
    ):
        """CollectPipeBase sends the selected image to the API Server.
        - app(Application): instance of actfw_core.Application
        - sender_task(AbstractSenderTask): instance of SenderTask
        - uncertainty(AbstractUncertainty): function to evaluate the uncertainty of an image
        - collect_frequency(int): frequency of sending data to the server. Unit is minutes.
        - batch_size(int): parameter for how many images to collect per send.
        - sspp_eps(float): trade-off parameter between accuracy and computational complexity.
                        The smaller the parameter, the higher the accuracy, but the increased computational complexity.
        - uncertainty_rate_lambda(float): weighting parameter to balance uncertainty/diversity
                        (diversity only if 0, uncertainty only if 1)
        - notifier(AbstractNotifier): message formatter to notify sending success/failure to Actcast
        """
        super().__init__()
        self.notifier = notifier

        sender_task.set_notifier(notifier)
        app.register_task(sender_task)

        self.select_task = SelectTask(
            uncertainty,
            sspp_eps,
            batch_size,
            sender_task,
            uncertainty_rate_lambda,
            collect_frequency
        )
        app.register_task(self.select_task)

    def interpret_inputs(self, inputs: D) -> Optional[DataDict]:
        """user has to implement this method in a subclass.
        this methods extracts data from this pipe inputs, and returns a DataDict.
        The returned DataDict's `image` must be PIL.Image, `feature_vector` must be of shape (N,)
        and `other_data` sub-dictionary should contain the
        necessary keys according to the chosen uncertainty function.
        This method may return None instead, in which case no data is passed
        to the selection process. (This can be used to filter the data to select from)
        """
        self.notifier.notify("interpret_inputs() is not implemented")
        raise NotImplementedError

    def proc(self, inputs: D) -> D:
        data_dict = self.interpret_inputs(inputs)
        if data_dict is not None:
            self.select_task.enqueue(data_dict)

        return inputs
