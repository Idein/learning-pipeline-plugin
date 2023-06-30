import time
from queue import Empty, Full, Queue
from typing import Generic, TypeVar

from actfw_core.task import Isolated

T_IN = TypeVar("T_IN")


class IsolatedTask(Generic[T_IN], Isolated):
    def __init__(self, inqueuesize: int = 0):
        """inqueuesize sets the max size of the incoming queue.
        Default: infinite size
        """
        super().__init__()
        self.in_queue: "Queue[T_IN]" = Queue(maxsize=inqueuesize)

    def enqueue(self, data: T_IN) -> bool:
        """Method to pass data to the Isolated Customer which will consume it in order.
        If queue is finite and full, skip data enqueue.

        returns True if the incoming element is added
        """
        try:
            self.in_queue.put_nowait(data)
        except Full:
            return False
        else:
            return True

    def run(self) -> None:
        """Implements Task.run() method
        Should not be called only by actfw_core.Application
        """
        while self.running:
            try:
                data = self.in_queue.get(timeout=1)
                self._proc(data)
            except Empty:
                time.sleep(1)
            except GeneratorExit:
                break

    def _proc(self, data: T_IN) -> None:
        """Isolated task process function.
        (called only internally)
        """
        raise NotImplementedError


class IsolatedTaskSingleBuffer(Generic[T_IN], IsolatedTask[T_IN]):
    def __init__(self, overwrite: bool = True):
        """overwrite parameter sets the behavior of enqueueing a full queue:
        with overwrite:
            the oldest element is discarded and the incoming element is added
        without overwrite:
            the incoming element is discarded
        """
        super().__init__(1)
        self._overwrite = overwrite

    def enqueue(self, data: T_IN) -> bool:
        """Put the incoming data in the queue, depending on the overwrite parameter
        see __init__ docstring for details.

        returns True if the incoming element is added
        """
        if self._overwrite:
            try:
                _ = self.in_queue.get_nowait()
            except Empty:
                pass

        return super().enqueue(data)
