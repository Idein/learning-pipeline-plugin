import bisect
import copy
import sys
from typing import Dict, List, Optional, cast, Iterable
from typing_extensions import Literal, TypedDict  # type: ignore

import numpy as np

from .submodular_utils import AbstractSubmodular, T, IndexedData
from .type_helper import DataSample


SimMetric = Literal["inner_product", "l1", "l2", "jsdiv"]
DivType = Literal["determinant", "mindistsum"]


class PreprocessParams(TypedDict, total=False):
    pre_norm: bool
    post_norm: bool
    sigma: float
    top_n: int
    rp_dims: int


def compute_logdet_diversity(matrix: np.ndarray, alpha=1.) -> float:
    """compute diversity d(S) = log det(I+aM_S)
    Arguments:
        matrix {ndarray} -- similarity matrix (S)
        alpha {float} -- coeff
    """
    m = np.eye(matrix.shape[0]) + alpha * matrix
    _, logdet = np.linalg.slogdet(m)
    return logdet


def compute_mindistsum_diversity(matrix: np.ndarray) -> float:
    """compute diversity d(S) = sum(min(d(v, v'))), {v, v'} in S
    Arguments:
        matrix {ndarray} -- similarity matrix (S)
    """
    matrix += np.eye(matrix.shape[0]) * np.max(matrix)  # exclude diag
    return np.sum(np.min(matrix, axis=1))


def compute_similarity_matrix(vectors: np.ndarray,
                              index_list: List[int],
                              sim_metric: SimMetric,
                              sim_buffer: Optional[Dict[int, Dict[int, float]]] = None) -> np.ndarray:
    """ compute similarity matrix
    Arguments:
        vectors {ndarray} --vectors to compute similarity matrix from. shape: (k, d)
        index_list {list of int} -- index of vector for memoization (sorted is assumed)
        sim_metric {str} -- similarity metric
        sim_buffer -- for memoization if not None
    Return:
        similarity matrix -- shape: (k, k)
    """
    k = len(vectors)

    if sim_metric == 'inner_product':
        def sim(v1, v2):
            return v1 @ v2
    elif sim_metric in ['l1', 'l2', 'jsdiv']:
        # distance-base: sim(v1, v2) = exp(-dist(v1, v2))
        if sim_metric == 'l1':
            def dist(v1, v2):
                return np.linalg.norm(v1 - v2, ord=1)
        elif sim_metric == 'l2':
            def dist(v1, v2):
                return np.linalg.norm(v1 - v2, ord=2)
        elif sim_metric == 'jsdiv':
            def dist(v1, v2):
                return np.sqrt(0.5*np.sum(v1*np.log2(2*v1/(v1+v2)) + v2*np.log2(2*v2/(v1+v2))))

        def sim(v1, v2):
            return np.exp(-dist(v1, v2))
    else:
        raise NotImplementedError('unknown sim_metric: {}'.format(sim_metric))

    # Force disable memoization for inner_product
    if sim_metric == 'inner_product':
        matrix = vectors @ vectors.T
        matrix = np.triu(matrix)

    # No memoization
    elif sim_buffer is None:
        matrix = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                matrix[i, j] = sim(vectors[i], vectors[j])

    # Memoization
    else:
        matrix = np.zeros((k, k))
        for i, idx_i in enumerate(index_list):
            if idx_i not in sim_buffer:
                sim_buffer[idx_i] = {}
            for j, idx_j in enumerate(index_list[i:], i):
                if idx_j in sim_buffer[idx_i]:
                    matrix[i, j] = sim_buffer[idx_i][idx_j]
                else:
                    val = sim(vectors[i], vectors[j])
                    sim_buffer[idx_i][idx_j] = val
                    matrix[i, j] = val

    matrix += np.triu(matrix, 1).T  # symmetric matrix
    return matrix


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1).reshape(-1, 1)
    norms[norms == 0] = 1
    return vectors/norms


def process_vectors(vector_list: List[np.ndarray],
                    index_list: List[int],
                    preprocess_buffer: Optional[Dict[int, np.ndarray]] = None,
                    rp_matrix: Optional[np.ndarray] = None,
                    params: Optional[PreprocessParams] = None) -> np.ndarray:
    """processing feature vector
    Arguments:
        vector_list {list of ndarray} -- vectors to compute similarity matrix from
        index_list {list of int} -- index of vector for memoization
        preprocess_buffer {dict} -- for memoization if not None
        rp_matrix {ndarray} -- matrix for random projection (D-dims -> d-dims)
        params {dict} -- parameters of preprocessing
            pre_norm {bool} -- normalize before projection
            post_norm {bool} -- normalize after projection
            sigma {float} -- if not None, feature vector v = exp(v*sigma)
    """
    vector_list = copy.deepcopy(vector_list)

    if params is None:
        params = {}

    pre_norm = params.get('pre_norm', False)
    post_norm = params.get('post_norm', False)
    sigma = params.get('sigma')
    top_n = params.get('top_n')

    if preprocess_buffer is not None:
        # memoization
        unbuffered_indices = []
        for i, ind in enumerate(index_list):
            if ind in preprocess_buffer:
                vector_list[i] = preprocess_buffer[ind]
            else:
                unbuffered_indices.append(i)
    else:
        unbuffered_indices = list(range(len(vector_list)))  # all

    if len(unbuffered_indices) > 0:
        unbuffered_vectors = np.array([vector_list[i] for i in unbuffered_indices])

        if pre_norm:
            unbuffered_vectors = normalize(unbuffered_vectors)

        if rp_matrix is not None:  # (D, d)
            unbuffered_vectors = unbuffered_vectors @ rp_matrix  # (n, d)

        if sigma is not None:
            unbuffered_vectors = np.exp(unbuffered_vectors*sigma)

        if top_n is not None:
            _vec = np.zeros(unbuffered_vectors.shape)
            _indices = np.argsort(-unbuffered_vectors, axis=1)[:, :top_n]  # indices of top-n values in each vector
            for i, _ind in enumerate(_indices):
                _vec[i][_ind] = unbuffered_vectors[i][_ind]
            unbuffered_vectors = _vec

        if post_norm:
            unbuffered_vectors = normalize(unbuffered_vectors)

        # memo
        if preprocess_buffer is not None:
            for v, ind in zip(unbuffered_vectors, np.array(index_list)[unbuffered_indices]):
                preprocess_buffer[ind] = v

        for v, i in zip(unbuffered_vectors, unbuffered_indices):
            vector_list[i] = v

    return np.array(vector_list)


class SubmodularDiversity(AbstractSubmodular[DataSample]):

    def __init__(self,
                 div_type: DivType = 'determinant',
                 sim_metric: SimMetric = 'inner_product',
                 alpha: float = 1.,
                 preprocess_params: Optional[PreprocessParams] = None,
                 memoization: bool = False,
                 preprocess_memoization: bool = False):
        if div_type == 'determinant':
            self.div_type = 'determinant'
        elif div_type == 'mindistsum':
            self.div_type = 'mindistsum'
        else:
            raise NotImplementedError('unknown div_type: {}'.format(div_type))

        self.sim_metric = sim_metric
        self.alpha = alpha
        self.preprocess_params = {} if preprocess_params is None else preprocess_params

        # random projection
        self.rp_dims = self.preprocess_params.get('rp_dims')
        self.rp_matrix: Optional[np.ndarray] = None  # (D, rp_dims)

        # memoization
        self.memoization = memoization
        self.sim_buffer = {} if memoization else None  # type: Optional[Dict[int, Dict[int, float]]]
        self.preprocess_buffer = {} if preprocess_memoization else None  # type: Optional[Dict[int, np.ndarray]]
        self._single_el_size = None  # type: Optional[int]

    def reset(self) -> None:
        self.sim_buffer = {}
        self.preprocess_buffer = {}

    def buffer_size(self) -> int:
        sim_buffer_size = 0
        if self.sim_buffer is not None:
            for el in self.sim_buffer.values():
                for fl in el.values():
                    sim_buffer_size += sys.getsizeof(fl)

        preprocess_buffer_size = 0
        if self.preprocess_buffer is not None:
            preprocess_buffer_size = len(self.preprocess_buffer)
            if preprocess_buffer_size > 0:
                if self._single_el_size is None:
                    arr = next(val for val in self.preprocess_buffer.values())
                    self._single_el_size = arr.nbytes
                preprocess_buffer_size *= cast(int, self._single_el_size)

        return sim_buffer_size + preprocess_buffer_size

    # for memoization
    def clear_indices(self, indices: Iterable[int]) -> None:
        if self.memoization and self.sim_metric != 'inner_product':
            assert self.sim_buffer is not None
            # 1 clear rows
            for idx in indices:
                del self.sim_buffer[idx]
                if self.preprocess_buffer is not None:
                    # self.preprocess_buffer.pop(idx, None)
                    del self.preprocess_buffer[idx]
            # 2 clear columns
            to_del = sorted(indices)
            s_keys = sorted(self.sim_buffer.keys())
            if len(to_del) > 0:
                to_del = to_del[bisect.bisect(to_del, s_keys[0]):]
            while len(to_del) > 0 and len(s_keys) > 0:
                i = bisect.bisect(s_keys, to_del[0])
                for idx in s_keys[:i]:
                    dico = self.sim_buffer[idx]
                    for to_del_idx in to_del:
                        # dico.pop(to_del_idx, None)  # remove unused index
                        del dico[to_del_idx]
                s_keys = s_keys[i:]
                if len(s_keys) > 0:
                    to_del = to_del[bisect.bisect(to_del, s_keys[0]):]
        elif self.preprocess_buffer is not None:
            for idx in indices:
                del self.preprocess_buffer[idx]
        else:
            pass

    def __call__(self, inputs_list: "List[IndexedData[DataSample]]") -> float:
        vector_list = [inputs[1].feature_vector for inputs in inputs_list]
        index_list = [inputs[0] for inputs in inputs_list]

        if self.rp_dims is not None and self.rp_matrix is None:
            self.rp_matrix = np.random.normal(0, 1, (vector_list[0].shape[0], self.rp_dims))

        vectors = process_vectors(vector_list, index_list, self.preprocess_buffer,
                                  self.rp_matrix, self.preprocess_params)

        similarity_matrix = compute_similarity_matrix(vectors, index_list, self.sim_metric,
                                                      self.sim_buffer)

        if self.div_type == 'determinant':
            return compute_logdet_diversity(similarity_matrix, alpha=self.alpha)
        elif self.div_type == 'mindistsum':
            return compute_mindistsum_diversity(similarity_matrix)
        else:
            raise NotImplementedError('unknown div_type: {}'.format(self.div_type))
