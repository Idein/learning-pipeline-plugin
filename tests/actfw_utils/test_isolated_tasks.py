import time
from typing import List

from learning_pipeline_plugin.actfw_utils import (IsolatedTask,
                                                  IsolatedTaskSingleBuffer)


class PrintIsolatedTask(IsolatedTask):
    def _proc(self, data: str) -> None:
        print(data)


class PrintSingleBuffer(IsolatedTaskSingleBuffer):
    def _proc(self, data: str) -> None:
        time.sleep(0.1)
        print(data)


def _enqueue_helper(task: IsolatedTask, inputs: List[str], expected_success: List[bool]) -> None:
    task.start()
    for i, b in zip(inputs, expected_success):
        assert task.enqueue(i) == b
        time.sleep(0.01)
    time.sleep(1)
    task.stop()
    task.join()


def test_simple_isolated(capfd):
    task = PrintIsolatedTask()
    inputs = ["a", "b", "c"]
    _enqueue_helper(task, inputs, [True]*3)

    out, _ = capfd.readouterr()
    assert out.splitlines() == inputs


def test_single_buffer_overwrite(capfd):
    overwrite_task = PrintSingleBuffer(overwrite=True)

    inputs = ["a", "b", "c"]
    _enqueue_helper(overwrite_task, inputs, [True, True, True])

    # overwrite sets output as True, but
    # middle input "b" has been overwritten during handling of "a"
    out, _ = capfd.readouterr()
    assert out.splitlines() == ["a", "c"]


def test_single_buffer(capfd):
    drop_task = PrintSingleBuffer(overwrite=False)

    inputs = ["a", "b", "c"]
    _enqueue_helper(drop_task, inputs, [True, True, False])

    # last input "c" has been dropped during handling of "a"
    out, _ = capfd.readouterr()
    assert out.splitlines() == ["a", "b"]
