from array import array
from typing import Any, Tuple, Callable, Dict, Iterable, List, Optional, Generic, TypeVar
from typing_extensions import TypedDict  # type: ignore


T = TypeVar("T")

# TODO: Python3.7 does not support TypedDict with Generic.
# Change to TypedDict when supporting python 3.8
IndexedData = Tuple[int, T]


class AbstractSubmodular(Generic[T]):

    def reset(self) -> None:
        pass

    def buffer_size(self) -> int:
        return 0

    def clear_indices(self, indices: Iterable[int]) -> None:
        pass

    def __call__(self, inputs_list: "List[IndexedData]") -> float:
        raise NotImplementedError()


class SubmodularPolicy(Generic[T], AbstractSubmodular[T]):

    def __init__(self, policy: Callable[[T], float]):
        self.policy = policy  # policy is callable and takes only kwargs
        self.buffer = array('f', (-1 for _ in range(100)))

    def reset(self):
        self.buffer = array('f', (-1 for _ in range(100)))

    def buffer_size(self) -> int:
        return self.buffer.buffer_info()[1] * self.buffer.itemsize

    def __call__(self, inputs_list: "List[IndexedData]") -> float:
        res: float = 0
        for inputs in inputs_list:
            idx, data = inputs
            while idx >= len(self.buffer):
                self.buffer.extend([-1]*100)
            val = self.buffer[idx]
            if val < 0:
                val = self.policy(data)
                self.buffer[idx] = val
            res += val
        return res


class SubmodularLC(Generic[T], AbstractSubmodular[T]):

    def __init__(self, submodular_list: List[AbstractSubmodular[T]], coeffs: Optional[List[float]] = None):
        self.submod_list = submodular_list
        if coeffs is None:
            self.coeffs = [1.]*len(submodular_list)
        else:
            self.coeffs = coeffs

    def clear_indices(self, indices: Iterable[int]) -> None:
        for sub_func in self.submod_list:
            sub_func.clear_indices(indices)

    def reset(self) -> None:
        for sub_func in self.submod_list:
            sub_func.reset()

    def buffer_size(self) -> int:
        return sum(sub_func.buffer_size() for sub_func in self.submod_list)

    def __call__(self, inputs_list: "List[IndexedData]") -> float:
        res = 0.
        for sub_f, coeff in zip(self.submod_list, self.coeffs):
            res += sub_f(inputs_list) * coeff
        return res
