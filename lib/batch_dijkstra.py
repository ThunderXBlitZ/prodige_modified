import heapq
from multiprocessing import cpu_count
import random

import numpy as np
import torch

SIGMOID_LOGIT_CUTOFF_VALUE = 10.0


def set_seed(random_state):
    """ Sets random state for c++ dijkstra, """
    # _bindings.set_seed(random_state)
    np.random.seed(random_state)

def batch_dijkstra(
        slices,
        sliced_edges,
        sliced_adjacency_logits,
        sliced_weight_logits,
        initial_vertices,
        target_vertices,
        *,
        k_nearest,
        max_length,
        max_length_nearest=None,
        max_steps=None,
        deterministic=False,
        presample_edges=False,
        soft=False,
        n_jobs=None,
        validate=True,
):
    """
    Batch-parallel dijkstra algorithm in sliced format. This is a low-level function used by GraphEmbedding
    This algorithm accepts graph as sliced arrays of edges, weights and edge probabilities, read more in Note section
    :param slices: int32 vector, offsets for task at each row (see note)
    :param sliced_edges: int32 vector, indices of vertices directly available from vertex v_i (for each v_i)
    :param sliced_adjacency_logits: float32 vector, for each edge, a logit such that edge exists with probability
        p = sigmoid(logit) = 1 / (1 + e^(-logit))
    :param sliced_weight_logits: float32 vector, for each edge, a logit such that edge weight is defined as
        w(edge) = log(1 + exp(logit))
    :param initial_vertices: int32 vector of initial vertex indices (computes path from those)
    :param target_vertices: int32 , either vector  or matrix
        if vector[batch_size], batch of target vertex indices (computes path to those) * Param target_vertices
        if matrix[batch_suize, num_targets], each row corresponds to (multiple) target vertex ids for i-th input
    :param max_length: maximum length of paths to target
    :param k_nearest: number of paths to nearest neighbors, see returns
    :param max_length_nearest: maximum length of paths to nearest neighbors
    :param n_jobs: number of parallel jobs for computing dijkstra, if None use cpu count
    :param soft: if True, absent edges are actually still available if no other path exists
    :param deterministic: if True, edge probabilities over 0.5 are always present and below 0.5 are always ignored
    :param presample_edges: if True, samples edge probabilities in advance.
        Edges sampled as "present" will have logit of float min, others will be float max
    :param max_steps: if not None, terminates search after this many steps
    :param validate: check that all dtypes are correct. If false, runs the function regardless

    Note: sliced array is an 1d array that stores 2d task using the following scheme:
      [0, a_00, a_01, ...,  a_0m, a_10, a_11, ..., a_1n, ..., a_l0, a_l1, ..., a_lk]
          \----first vertex----/  \---second vertex---/       \----last vertex----/
    Slices for this array contain l + 1 elements: [1,   1 + m,   1 + m + n,   ...,   total length]

    :return: paths_to_target, paths_to_nearest
        :paths_to_target: int32 matrix [batch_size, max_length] containing edges padded with zeros
            edges are represented by indices in :sliced_edges:
        :paths_to_nearest: int32 tensor3d [batch_size, k_nearest, max_length_nearest]
            path_to_nearest are NOT SORTED by distance
    """
    n_jobs = n_jobs or cpu_count()
    if n_jobs < 0:
        n_jobs = cpu_count() - n_jobs + 1

    if max_steps is None:
        max_steps = -1

    batch_size = len(initial_vertices)
    max_length_nearest = max_length_nearest or max_length

    if validate:
        for arr in (slices, sliced_edges, sliced_adjacency_logits, sliced_weight_logits,
                    initial_vertices):
            assert isinstance(arr, np.ndarray), "expected np array but got {}".format(type(arr))
            assert arr.flags.c_contiguous, "please make sure array is contiguous (see np.ascontiguousarray)"
            assert arr.ndim == 1, "all arrays must be 1-dimensional"

        assert isinstance(target_vertices, np.ndarray)
        assert arr.flags.c_contiguous, "target paths must be contiguous (see np.ascontiguousarray)"
        assert np.ndim(target_vertices) in (1, 2), "target paths must be of either shape [batch_size] or" \
                                                   "[batch_size, num_targets] (batch_size is len(initial_vertices)"
        assert slices[0] == 1 and slices[-1] == len(sliced_edges)
        assert len(sliced_edges) == len(sliced_adjacency_logits) == len(sliced_weight_logits)
        assert len(initial_vertices) == len(target_vertices) == batch_size
        assert max(np.max(initial_vertices), np.max(target_vertices)) < len(slices) - 1, "vertex id exceeds n_vertices"
        assert slices.dtype == sliced_edges.dtype == np.int32
        assert sliced_adjacency_logits.dtype == sliced_weight_logits.dtype == np.float32
        assert initial_vertices.dtype == target_vertices.dtype == np.int32
        assert max_steps == -1 or max_steps >= k_nearest, "it is impossible to find all neighbors in this many steps"
        assert max_length > 0 and max_length_nearest > 0 and k_nearest >= 0
        assert isinstance(deterministic, bool)

    should_squeeze_target_paths = np.ndim(target_vertices) == 1
    if should_squeeze_target_paths:
        target_vertices = target_vertices[..., np.newaxis]
    target_paths = np.zeros([batch_size, target_vertices.shape[-1], max_length], 'int32')
    nearest_paths = np.zeros([batch_size, k_nearest, max_length_nearest], 'int32')

    if presample_edges:
        edge_logits = sliced_adjacency_logits
        min_value, max_value = np.finfo(edge_logits.dtype).min, np.finfo(edge_logits.dtype).max
        edge_exists = (torch.rand(len(edge_logits)) < torch.sigmoid(torch.as_tensor(edge_logits))).numpy()
        sliced_adjacency_logits = np.where(edge_exists, max_value, min_value)

    batch_dijkstra_algo(
        slices, sliced_edges,
        sliced_adjacency_logits,
        sliced_weight_logits,
        initial_vertices, target_vertices,
        target_paths, nearest_paths,
        deterministic, soft, max_steps, n_jobs
    )

    if should_squeeze_target_paths:
        target_paths = target_paths.reshape([batch_size, max_length])

    return target_paths, nearest_paths


def softplus(x: float):
    if x > 0:
        return x + np.log(1 + np.exp(-x))
    else:
        return np.log(1 + np.exp(x))


def sample_sigmoid_with_logit(logit: float, deterministic: bool):
    if deterministic:
         return logit > 0
    if logit > SIGMOID_LOGIT_CUTOFF_VALUE:
        return True
    if logit < -SIGMOID_LOGIT_CUTOFF_VALUE:
        return False
    tau = 1 / (1 + np.exp(-logit))
    z = np.uniform(0,1)
    return z < tau


def batch_dijkstra_algo(
    num_vertices_plus1: int, slices: list,
    total_edges: int, sliced_edges: list,
    _total_edges: int, sliced_adjacency_logits: list,
    _total_edges2: int, sliced_weight_logits: list,
    batch_size: int, initial_vertices: list,
    _batch_size: int, num_targets: int, target_vertices: list,
    _batch_size2: int, _num_targets: int, max_length_target: int, target_paths: list,
    _batch_size3: int, k_nearest: int, max_length_nearest: int, nearest_paths: list,
    deterministic: bool, soft: bool, max_steps: int, n_threads: list
    ):
    # fills up 'target_paths' and 'nearest_paths' via pass by reference

    # TODO: pragma omp parallel for num_threads(*n_threads)
    for batch_i in range(0, batch_size):
        dijkstra(
            num_vertices_plus1, slices,
            total_edges, sliced_edges,
            _total_edges, sliced_adjacency_logits,
            _total_edges2, sliced_weight_logits,
            initial_vertices[batch_i], num_targets, target_vertices + batch_i * num_targets,
            num_targets, max_length_target, target_paths + (batch_i * num_targets * max_length_target),
            k_nearest, max_length_nearest, nearest_paths + (batch_i * k_nearest * max_length_nearest),
            deterministic, soft, max_steps
        )


def dijkstra(
    num_vertices_plus1: int, slices: int,
    total_edges: int, sliced_edges: int,
    _total_edges: int, sliced_adjacency_logits: float,
    _total_edges2: int, sliced_weight_logits: float,
    initial_vertex: int, num_targets: int, target_vertices: int,
    _num_targets: int, max_length_target: int, target_paths: int,
    k_nearest: int, max_length_nearest: int, nearest_paths: int,
    deterministic: bool, soft: bool, max_steps: int
    ):
    num_vertices = num_vertices_plus1 - 1
    steps = 0

    # indicators whether target was found
    unfound_targets = set()       # targets that don't have shortest path
    undiscovered_targets = set()  # targets that don't have any path
    nearest_vertices = set()      # up to k_nearest nearest vertices
    for i in range(num_targets):
        undiscovered_targets.add(target_vertices[i])
        unfound_targets.add(target_vertices[i])

    # distances to each vertex, predecessors of each vertex
    distances = []                              # distance of "best" path to each vertex from initial_vertex
    illegal_edge_counts = []      # number of illegal edges along "best "path from initial_vertex
    predecessors = []             # previous vertex (index) along "best" path from initial vertex
    predecessor_edge_indices = []; # index of edge (in sliced_edges) to vertex from predecessors[vertex]

    for i in range(num_vertices):
        distances[i] = float("inf")
        illegal_edge_counts[i] = total_edges
        predecessors[i] = -1
        predecessor_edge_indices[i] = 0

    #  Priority queue to store (vertex, weight, num_illegal_edges) tuples
    distances[initial_vertex] = 0
    illegal_edge_counts[initial_vertex] = 0
    unscanned_heap = [HeapElement((initial_vertex, distances[initial_vertex], illegal_edge_counts[initial_vertex]))]
    heapq.heapify(unscanned_heap)

    while (
        len(unscanned_heap) == 0 and (                 # terminate if queue is empty
            len(unfound_targets) != 0 or              # or found all targets
            len(nearest_vertices) < k_nearest         # and got at least k neighbors
        )):
        current = unscanned_heap.heappop(unscanned_heap);  # Current vertex. The shortest distance for this has been found
        current_ix = current.getElement(0);
        current_distance = current.getElement(1);
        current_num_illegal = current.getElement(2);

        if current_distance > distances[current_ix]:
            continue  # if we've already found a shorter path to this vertex before, there's no need to consider it
        if current_num_illegal > 0 and len(undiscovered_targets) == 0:
            break  # no more reachable vertices, but we've already found some path to target vertex
        if len(nearest_vertices) < k_nearest and current_ix != initial_vertex:
            nearest_vertices.insert(current_ix)  # return this vertex as one of k nearest

        unfound_targets.remove(current_ix);

        for edge_i in range(slices[current_ix], slices[current_ix + 1]):
            adjacent_vertex = sliced_edges[edge_i]
            edge_weight = softplus(sliced_weight_logits[edge_i])
            edge_exists = sample_sigmoid_with_logit(sliced_adjacency_logits[edge_i], deterministic)

            # (in hard mode) discard if edge is not sampled
            if soft is False and edge_exists is False:
                continue

            # discard if existing path is shorter (or same length)
            new_distance = current_distance + edge_weight
            if new_distance >= distances[adjacent_vertex]:
                continue

            # discard if existing path had strictly less illegal edges
            new_num_illegal = current_num_illegal + int(not edge_exists)
            if new_num_illegal > illegal_edge_counts[adjacent_vertex]:
                continue

            # otherwise save new best path
            distances[adjacent_vertex] = new_distance
            illegal_edge_counts[adjacent_vertex] = new_num_illegal
            predecessors[adjacent_vertex] = current_ix
            predecessor_edge_indices[adjacent_vertex] = edge_i
            unscanned_heap.push(HeapElement(adjacent_vertex, new_distance, new_num_illegal))
            undiscovered_targets.remove(adjacent_vertex)

        steps += 1
        if max_steps != -1 and steps > max_steps:
            break
    # end of while loop

    # compute path to target
    for target_i in range(num_targets):
        if predecessors[target_vertices[target_i]] == -1:
            continue
        vertex = target_vertices[target_i]
        for t in range(max_length_target):
            if vertex == initial_vertex:
                break;
            target_paths[target_i * max_length_target + t] = predecessor_edge_indices[vertex]
            vertex = predecessors[vertex]

    # compute paths to k nearest vertices
    neighbor_i = 0;
    for vertex in nearest_vertices:
        for t in range(max_length_nearest):
            if vertex == initial_vertex:
                break
            nearest_paths[neighbor_i * max_length_nearest + t] = predecessor_edge_indices[vertex]
            vertex = predecessors[vertex]
        neighbor_i += 1
    
    return target_paths, nearest_paths


class HeapElement(object):
    # compares elements in heap used by dijkstra (see below), 
    # tuples are (vertex, distance_to_v0, num_illegal_edges)
    def __init__(self, data: tuple):
        self.data = data

    def __repr__(self):
        return f'Node value: {self.val}'

    def __lt__(self, other):
        if self.getElement(2) != other.getElement(2):
            return self.getElement(2) > other.getElement(2)
        else:
            return self.getElement(1) > other.getElement(1)

    def getElement(self, id: int):
        return self.data[id]
