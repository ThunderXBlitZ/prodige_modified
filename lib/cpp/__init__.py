"""
Does all sorts of dark magic in order to build/import c++ bfs
"""
import os
import os.path as osp
import random

import setuptools.sandbox
from multiprocessing import cpu_count
from lib import cpp
import numpy as np
import torch

package_abspath = osp.join(*osp.split(osp.abspath(__file__))[:-1])
if not os.path.exists(osp.join(package_abspath, "_bindings.so")):
    # try build _bfs.so
    workdir = os.getcwd()
    try:
        os.chdir(package_abspath)
        setuptools.sandbox.run_setup(
            osp.join(package_abspath, "setup.py"), ["clean", "build"]
        )
        os.system(
            "cp {}/build/lib*/*.so {}/_bindings.so".format(package_abspath, package_abspath)
        )
        assert os.path.exists(osp.join(package_abspath, "_bindings.so"))
    finally:
        os.chdir(workdir)

from . import _bindings
_bindings.set_seed(random.randint(0, 2 ** 16))


def set_seed(random_state):
    """ Sets random state for c++ dijkstra, """
    _bindings.set_seed(random_state)


def batch_dijkstra(
        slices,
        sliced_edges,
        sliced_adjacency_logits,
        sliced_weight_logits,
        initial_vertices,
        target_vertices,
        *,
        k_nearest=0,
        max_length=10,
        max_length_nearest=None,
        max_steps=-1,
        deterministic=True,
        soft=True,
        n_jobs=-1
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
    :param max_steps: terminates search after this many steps

    :return: paths_to_target, paths_to_nearest
        :paths_to_target: int32 matrix [batch_size, max_length] containing edges padded with zeros
            edges are represented by indices in :sliced_edges:
        :paths_to_nearest: int32 tensor3d [batch_size, k_nearest, max_length_nearest]
            path_to_nearest are NOT SORTED by distance
    """
    n_jobs = cpu_count() if n_jobs < 0 else n_jobs

    batch_size = len(initial_vertices)
    max_length_nearest = max_length_nearest or max_length

    should_squeeze_target_paths = np.ndim(target_vertices) == 1
    if should_squeeze_target_paths:
        target_vertices = target_vertices[..., np.newaxis]
    target_paths = np.zeros([batch_size, target_vertices.shape[-1], max_length], 'int32')
    nearest_paths = np.zeros([batch_size, k_nearest, max_length_nearest], 'int32')

    _bindings.batch_dijkstra(
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
