from typing import Generic, Optional, TypeVar

from actfw_core import Application
from actfw_core.task import Pipe

from .algorithms.type_helper import DataDict
from .algorithms.uncertainty import AbstractUncertainty
from .notifier import AbstractNotifier, Notifier
from .select_task import SelectTask
from .sender_task import SenderTask, UserMetadata

D = TypeVar("D")


class CollectPipeBase(Generic[D], Pipe[D, D]):
    def __init__(
        self,
        app: Application,
        uncertainty: AbstractUncertainty,
        collect_frequency: int,
        batch_size: int,
        sspp_eps: float = 0.05,
        sspp_lambda: float = 1.0,
        metadata: UserMetadata = {},
        send_inqueuesize: int = 0,
        endpoint_url: str = "api.autolearner.actcast.io",
        notifier: AbstractNotifier = Notifier(),
    ):
        """CollectPipeBase sends the selected image to the API Server.
        - app(Application): instance of actfw_core.Application
        - uncertainty(AbstractUncertainty): function to evaluate the uncertainty of an image
        - collect_frequency(int): frequency of sending data to the server. Unit is minutes.
        - batch_size(int): parameter for how many images to collect per send.
        - sspp_eps(float): trade-off parameter between accuracy and computational complexity.
                        The smaller the parameter, the higher the accuracy, but the increased computational complexity.
        - sspp_lambda(float): weighting parameter for diversity when uncertainty is 1.0
        - metadata(UserMetadata): JSON-like data that will be stored with the image
        - send_inqueuesize(int): size of the sending queue (default: 0 (no limit))
        - endpoint_root(str): endpoint root of the lp API server
        - notifier(AbstractNotifier): message formatter to notify sending success/failure to Actcast
        """
        super().__init__()
        self.notifier = notifier

        sender_task = SenderTask(endpoint_url, self.notifier, metadata, inqueuesize=send_inqueuesize)
        app.register_task(sender_task)

        self.select_task = SelectTask(
            uncertainty,
            sspp_eps,
            batch_size,
            sender_task,
            sspp_lambda,
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
