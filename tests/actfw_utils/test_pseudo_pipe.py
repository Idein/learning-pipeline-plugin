import itertools
import pytest
import time
from typing import List, Optional
import threading

from actfw_core.application import Application
from actfw_core.task import Consumer, Producer

from learning_pipeline_plugin.actfw_utils import PseudoPipe


class StringProducer(Producer[str]):
    def __init__(self, msg: str) -> None:
        super().__init__()
        self.msg = msg

    def proc(self) -> str:
        return self.msg


class PrintConsumer(Consumer[str]):
    def __init__(self, msg_prefix: str = "", lock: Optional[threading.Lock] = None) -> None:
        super().__init__()
        self.prefix = msg_prefix
        self.print_lock = lock if lock is not None else threading.Lock()

    def proc(self, i: str) -> None:
        with self.print_lock:
            print(f"{self.prefix}{i}")


class ThreadedApplication(Application):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread = None

    def run(self) -> None:
        self.thread = threading.Thread(target=super().run)
        self.thread.start()

    def stop(self) -> None:
        assert self.thread is not None
        super().stop()
        self.thread.join(10)


def test_pseudo_pipe(capfd):
    msg = "hello world"
    app = ThreadedApplication()

    prod = StringProducer(msg)
    cons = PrintConsumer()
    pp = PseudoPipe()

    app.register_task(prod)
    app.register_task(cons)

    # pseudo pipe cannot be registered
    with pytest.raises(TypeError):
        app.register_task(pp)

    prod.connect(pp)
    pp.connect(cons)
    app.run()
    time.sleep(0.1)
    app.stop()

    # pseudo pipe only pass data through
    out, _ = capfd.readouterr()
    assert msg in out


def test_pseudo_pipe_multi_out(capfd):
    app = ThreadedApplication()
    print_lock = threading.Lock()
    num = 3

    prods = []
    for i in range(num):
        prods.append(StringProducer(str(i)))
    cons = []
    for i in range(num):
        cons.append(PrintConsumer(str(i), lock=print_lock))

    pp = PseudoPipe()

    for p in prods:
        app.register_task(p)
        p.connect(pp)
    for c in cons:
        app.register_task(c)
        pp.connect(c)

    app.run()
    time.sleep(0.1)
    app.stop()

    out, _ = capfd.readouterr()
    for t in itertools.product(range(num), repeat=2):
        string = "".join(str(el) for el in t)
        assert string in out
