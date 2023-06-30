from typing import Generic

from actfw_core.task.consumer import _PadOut
from actfw_core.task.pipe import T_IN, T_OUT, _ConsumerMixin, _ProducerMixin


class PseudoPipe(Generic[T_IN], _ConsumerMixin[T_IN]):
    """A pipe that connects its precedent and following tasks together
    e.g.
    ```python
    p = MyProducer(...)
    c = MyConsumer(...)
    pp = PseudoPipe()

    p.connect(c)
    # is the same as
    p.connect(pp)
    pp.connect(c)
    ```

    It is not a Task subclass, so it cannot be registered.
    """
    def __init__(self) -> None:
        super().__init__()
        self._connected = False

    def _is_running(self) -> bool:
        return False

    def _add_in_queue(self, q: _PadOut[T_IN]) -> None:
        assert not self._connected, "PseudoPipe is given input task after being connected to output"
        super()._add_in_queue(q)

    def connect(self, follow: _ConsumerMixin[T_IN]) -> None:
        self._connected = True
        assert isinstance(follow, _ConsumerMixin)
        assert len(self.in_queues) > 0, "PseudoPipe connected to nothing"
        for pad in self.in_queues:
            follow._add_in_queue(pad)
