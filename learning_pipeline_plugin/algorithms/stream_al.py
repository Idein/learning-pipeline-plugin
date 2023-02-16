from array import array
import random
from typing import List, Iterable, Any, Union, Tuple, Dict, Optional, Generic, TypeVar, Callable
from typing_extensions import TypedDict  # type: ignore
import numpy as np

from .submodular_utils import AbstractSubmodular, T, IndexedData


class AbstractStreamSelector(Generic[T]):

    def judge(self, inputs: T) -> None:
        # use policy to add it or not to the current in preparation batch
        raise NotImplementedError

    def get_selected_batch(self) -> List[T]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class MemoizationSieveStreamingPlusPlusSelector(Generic[T], AbstractStreamSelector[T]):

    def __init__(self, batch_size: int, f: AbstractSubmodular[T], eps: float,
                 filter_out_thresh: Optional[float] = None):
        super().__init__()
        self._batch_size = batch_size
        self.delta: float = 0  # Max value element we have seen so far.
        self.lower_bound: float = 0  # Lower bound on OPT (the optimal possible value for this function).
        self.s_t: Dict[float, List[int]] = {}
        # Dictionary of thresholds (based on possible guesses for OPT).
        # Each T[guess] is a list S of elements selected for that threshold.
        self.vals: Dict[float, float] = {}
        # Values of set held for each threshold (so we don't have to keep recomputing it).
        self.idx_to_data: Dict[int, IndexedData] = {}  # * for memoization
        self.buffer_use = array('H', (0 for _ in range(100)))
        self.submodular_f = f
        self.eps = eps
        self.filter_out_thresh = filter_out_thresh

        self.current_index = 0

    def get_selected_batch(self) -> List[T]:
        bestValue: float = 0  # Set default values.
        bestSet = []
        for threshold in self.s_t:  # find the best set amongst all still-valid thresholds.
            set_of_data: List[IndexedData] = [self.idx_to_data[idx] for idx in self.s_t[threshold]]
            val = self.vals[threshold]
            if val > bestValue:
                bestValue = val
                bestSet = set_of_data

        return [element[1] for element in bestSet]

    def reset(self) -> None:
        self.delta = 0
        self.lower_bound = 0
        self.s_t = {}
        self.idx_to_data = {}
        self.buffer_use = array('H', (0 for _ in range(100)))
        self.vals = {}
        self.current_index = 0
        self.submodular_f.reset()  # in case submodular function uses memoization

    def judge(self, inputs: T) -> None:
        i_inputs: IndexedData = (self.current_index, inputs)
        if self.current_index >= len(self.buffer_use):
            self.buffer_use.extend([0]*100)

        # Update max value element we have seen so far, if appropriate.
        self.delta = max(self.delta, self.submodular_f([i_inputs]))
        # Update tau min (lower threshold based on guess for OPT).
        tau_min = max(self.lower_bound, self.delta)/(2*self._batch_size)

        self._updateThresholdsData(tau_min)  # Drop all thresholds that are below tauMin.

        add_key = False
        if self.filter_out_thresh is not None and self.filter_out_thresh >= self.submodular_f([i_inputs]):
            # skips sample
            pass
        else:
            for threshold in self.s_t:
                # If we have not already selected k elements in the set corresponding to this threshold:
                if len(self.s_t[threshold]) < self._batch_size:
                    newVal = self.submodular_f([self.idx_to_data[idx] for idx in self.s_t[threshold]]+[i_inputs])
                    # Calculate the marginal gain of the current element in the stream, relative to this set.
                    marginalGain = newVal - self.vals[threshold]
                    if marginalGain > threshold:
                        add_key = True
                        self.buffer_use[self.current_index] += 1
                        # If the marginalGain is above the required 'threshold', add the element to the set.
                        self.s_t[threshold].append(self.current_index)
                        self.vals[threshold] = newVal
                        self.lower_bound = max(self.lower_bound, newVal)  # Update the lower bound on OPT if necessary.
        if add_key:
            self.idx_to_data[self.current_index] = i_inputs
        else:
            del i_inputs
            del inputs
            self.submodular_f.clear_indices([self.current_index])
        self.current_index += 1

    def _updateThresholdsData(self, tau_min: float) -> None:
        if tau_min <= 0:
            return
        threshold = 1.0
        while threshold > tau_min/(1.0+self.eps):
            if threshold < self.delta and threshold not in self.s_t:
                self.s_t[threshold] = []
                self.vals[threshold] = 0.0
            threshold = threshold/(1.0+self.eps)

        threshold = 1.0+self.eps
        while threshold < self.delta:
            if threshold > tau_min and threshold not in self.s_t:
                self.s_t[threshold] = []
                self.vals[threshold] = 0.0
            threshold = threshold*(1.0+self.eps)

        toDelete = []
        for key in self.s_t:
            if key < tau_min:
                toDelete.append(key)

        unused_index = set()
        for key in toDelete:
            for idx in self.s_t[key]:
                self.buffer_use[idx] -= 1
                if self.buffer_use[idx] == 0:
                    unused_index.add(idx)
            del self.s_t[key]
            del self.vals[key]
        self.clear_indices(unused_index)

    def clear_indices(self, indices: Iterable[int]) -> None:
        # clear the elements from memory, they won't be used anymore
        self.submodular_f.clear_indices(indices)
        for idx in indices:
            del self.idx_to_data[idx]
