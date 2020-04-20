import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from collections import namedtuple
from itertools import chain
import scipy.stats

from .cpp import batch_dijkstra
from .utils import check_numpy, sliced_argmax, inverse_softplus, inverse_sigmoid
from collections import defaultdict
import scipy.sparse, scipy.sparse.csgraph
from sklearn.neighbors import NearestNeighbors


def initalize_prodige(X, knn_edges=64, random_edges=32, verbose=False, **kwargs):
    # Creates graph embedding from an object-feature matrix,
    # initialize weights with squared euclidian distances |x_i - x_j|^2_2

    # 2 types of edges:
    #   knn edges - connecting vertices to their nearest neighbors
    #   Random edges - connecting random pairs of vertices to get smallworld property

    # X: matrix of samples
    # See Section 3.3: Scalability
    # knn_edges: edges per vertex to X nearest neighbours via FAISS or sklearn
    # random_edges: X random edges per vertex for smallworld property
    # kwargs: other args sent to GraphEmbedding()
    # returns: Initialized GraphEmbedding

    num_vectors, vector_dim = X.shape
    X = np.require(X, dtype=np.float32, requirements=['C_CONTIGUOUS'])

    if verbose:
        print("Searching for nearest neighbors")
    try:
        from faiss import IndexFlatL2
        index = IndexFlatL2(vector_dim)
        index.add(X)
        neighbor_distances, neighbor_indices = index.search(X, knn_edges + 1)
    except ImportError:
        print("faiss not found, using slow knn instead")
        neighbor_distances, neighbor_indices = NearestNeighbors(n_neighbors=knn_edges + 1).fit(X).kneighbors(X)

    if verbose:
        print("Adding knn edges")
    edges_from, edges_to, distances = [], [], []
    for vertex_i in np.arange(num_vectors):
        for neighbor_i, distance in zip(neighbor_indices[vertex_i], neighbor_distances[vertex_i]):
            if vertex_i == neighbor_i: continue  # prevent loops
            if neighbor_i == -1: continue  # padding
            distance **= 0.5
            edges_from.append(vertex_i)
            edges_to.append(neighbor_i)
            distances.append(distance)

    if random_edges != 0:
        if verbose: print("Adding random edges")
        random_from = np.random.randint(0, num_vectors, num_vectors * random_edges)
        random_to = np.random.randint(0, num_vectors, num_vectors * random_edges)
        for vertex_i, neighbor_i in zip(random_from, random_to):
            if vertex_i != neighbor_i:
                distance = np.sum((X[vertex_i] - X[neighbor_i]) ** 2) ** 0.5
                edges_from.append(vertex_i)
                edges_to.append(neighbor_i)
                distances.append(distance)

    if verbose: print("Deduplicating edges")
    # remove duplicate edges and add them again at random
    unique_edges_dict = {}
    for from_i, to_i, distance in zip(edges_from, edges_to, distances):
        edge_iijj = int(from_i), int(to_i)
        edge_iijj = tuple(sorted(edge_iijj))
        unique_edges_dict[edge_iijj] = distance
    edges_iijj, distances = zip(*unique_edges_dict.items())
    edges_from, edges_to = zip(*edges_iijj)

    edges_from, edges_to, distances = map(np.asanyarray, [edges_from, edges_to, distances])
    if verbose:
        print("Total edges: {}, mean edges per vertex: {}, mean distance: {}".format(
            len(edges_from), len(edges_from) / float(num_vectors), np.mean(distances)
        ))
    return GraphEmbedding(edges_from, edges_to, weights=distances, **kwargs)


class GraphEmbedding(nn.Module):
    edges = namedtuple("edges", ["adjacent", "p_adjacent", "weights"])
    INF = torch.tensor(float('inf'), dtype=torch.float32)
    NEG_INF = -INF
    
    def __init__(self, edges_from, edges_to, *, weights=1.0, probs=0.9):
        # Initialize PRODIGE Graph embeddings
        # edges_from: an array of sources of all edges, int[num_edges]
        # edges_to: an array of destinations of all edges, int[num_edges]
        # weights: vector of weights per edge
        # probs: vector of probs per edge

        super().__init__()

        if np.ndim(weights) == 0:
            weights = np.full(edges_from.shape, float(weights), dtype=np.float32)
        if np.ndim(probs) == 0:
            probs = np.full(edges_from.shape, float(probs), dtype=np.float32)
        edges_from, edges_to, weights, probs = map(
            check_numpy, [edges_from, edges_to, weights, probs])

        num_vertices = max(edges_from.max(), edges_to.max()) + 1
        adjacency_logits = inverse_sigmoid(probs)
        weight_logits = inverse_softplus(weights)

        # Create graph as an adjacency list
        adjacency = [list() for _ in range(num_vertices)]
        edge_to_weight_logits, edge_to_adjacency_logits = {}, {}
        for from_i, to_i, adj_logit, weight_logit in zip(
                edges_from, edges_to, adjacency_logits, weight_logits):
            edge = (int(from_i), int(to_i))
            from_i, to_i = edge = tuple(sorted(edge))
            adjacency[from_i].append(to_i)
            edge_to_adjacency_logits[edge] = adj_logit
            edge_to_weight_logits[edge] = weight_logit

        adjacency = list(map(sorted, adjacency))
        edge_sources = list(chain(*([i] * len(adjacency[i]) for i in range(len(adjacency)))))
        edge_targets = list(chain(*adjacency))
        lengths = list(map(len, adjacency))

        # Compute slices and edge_indices
        self.edge_sources = np.array([0] + edge_sources, dtype=np.int32)
        self.edge_targets = np.array([0] + edge_targets, dtype=np.int32)
        self.slices = np.cumsum([1] + lengths).astype("int32")

        # Save Torch trainable matrices
        self.edge_adjacency_logits = nn.Parameter(torch.randn(len(self.edge_targets), 1), requires_grad=True)
        self.edge_weight_logits = nn.Parameter(torch.randn(len(self.edge_targets), 1), requires_grad=True)
        self.default_distance = nn.Parameter(torch.tensor(0.).view(1, 1), requires_grad=True)
        self.num_vertices, self.num_edges = num_vertices, len(self.edge_targets)
        
        # initialize and store weight and adjacency logits
        with torch.no_grad():
            flat_i = 1  # start from 1 to skip first "fake" edge
            for from_i, to_ix in enumerate(adjacency):
                for to_i in to_ix:
                    edge = (int(from_i), int(to_i))
                    self.edge_adjacency_logits.data[flat_i, 0] = float(edge_to_adjacency_logits[edge])
                    self.edge_weight_logits.data[flat_i, 0] = float(edge_to_weight_logits[edge])
                    flat_i += 1

        # Convert undirected edges to directed by duplicating them
        directed_edges_by_source = defaultdict(set)
        directed_edge_to_ix = {}

        for i, from_i, to_i in zip(np.arange(self.num_edges), self.edge_sources, self.edge_targets):
            if i == 0: continue
            directed_edges_by_source[from_i].add(to_i)
            directed_edges_by_source[to_i].add(from_i)
            directed_edge_to_ix[from_i, to_i] = i
            directed_edge_to_ix[to_i, from_i] = i

        directed_edges = [0]
        directed_slices = [1]
        directed_to_undirected_reorder = [0]
        for from_i in range(self.num_vertices):
            directed_edges_from_i = sorted(directed_edges_by_source[from_i])
            directed_slices.append(directed_slices[-1] + len(directed_edges_from_i))
            directed_edges.extend(directed_edges_from_i)
            directed_to_undirected_reorder.extend(
                [directed_edge_to_ix[from_i, to_i] for to_i in directed_edges_from_i])

        # Store graph as sliced array
        self.directed_edge_indices = np.array(directed_edges, dtype='int32')
        self.directed_slices = np.array(directed_slices, dtype='int32')
        self.reorder_undirected_to_directed = torch.as_tensor(directed_to_undirected_reorder, dtype=torch.int64)

    def _get_logits(self, sliced_logits, sliced_indices):
        sliced_indices = self.reorder_undirected_to_directed[sliced_indices]
        return F.embedding(sliced_indices, sliced_logits, sparse=True).view(*sliced_indices.shape)

    def _get_default_distance(self):
        return F.embedding(torch.zeros(1, dtype=torch.int64), self.default_distance, sparse=True).view([])

    def get_edges(self, vertex):
        begin_i, end_i = self.directed_slices[vertex], self.directed_slices[vertex + 1]
        edge_span = torch.arange(begin_i, end_i, dtype=torch.int64,
                                 device=self.edge_adjacency_logits.device)

        return self.edges(
            torch.as_tensor(self.directed_edge_indices[begin_i: end_i]), # adjacent vertex
            torch.sigmoid(self._get_logits(self.edge_adjacency_logits, edge_span)), # prob 
            F.softplus(self._get_logits(self.edge_weight_logits, edge_span)), # weight
        )

    def compute_l0_prior_penalty(self, batch_size, lambd=1.0):
        # Computes negative L0_prior = - mean(1 - P(edge))
        # lambd: scalar regularization coefficient, see https://arxiv.org/abs/1712.01312 for details
        # batch_size: averages over this many random edges
        # See Section 3.2: Sparsity

        regularized_indices = np.arange(1, len(self.directed_edge_indices))
        batch = torch.randint(0, len(regularized_indices), (batch_size,), device=self.edge_adjacency_logits.device)
        regularized_indices = torch.as_tensor(regularized_indices)[batch]
        p_keep_edge = torch.sigmoid(self._get_logits(self.edge_adjacency_logits, regularized_indices))
        return lambd * p_keep_edge.mean()

    def compute_pairwise_distances(self):
        # Computes distances between all pairs of points, returns distances matrix

        indices = np.arange(self.num_vertices)
        edge_threshold = 0.5

        edges = defaultdict(lambda: float('inf')) # format: {(from, to) -> weight}
        for ix in range(self.num_vertices):
            adjs, probs, weights = self.get_edges(ix)
            adjs = adjs[probs >= edge_threshold].detach().numpy()
            weights = weights[probs >= edge_threshold].detach().numpy()
            for (target, weight) in zip(adjs, weights):
                edges[int(ix), int(target)] = min(edges[int(ix), int(target)], weight)

        from_to, weights = zip(*edges.items())
        froms, tos = zip(*from_to)
        sparse_edges = scipy.sparse.coo_matrix((weights, (froms, tos)), shape=[self.num_vertices, self.num_vertices])
        distances = scipy.sparse.csgraph.dijkstra(sparse_edges, directed=False, indices=indices)
        default = float(self._get_default_distance().item())
        distances[np.isinf(distances)] = default
        return distances
    
    def forward(self, from_ix, to_ix, **parameters):
        # Overridden PyTorch class function, returns matrices for gradident calculation
        # Computes path from from_ix to to_ix
        # from_ix: a vector of initial vertice, length batch_size
        # to_ix: a vector of target vertices, length batch_size
        # parameters: see lib.cpp.batch_dijkstra
        # return: {target_paths & target_distances, logp_target_paths (for log derivative trick), 
        #       found_target, nearest_vertices, nearest_paths & nearest_distances}
        
        # set defaults
        parameters['deterministic'] = parameters.get('deterministic', not self.training)
        parameters['k_nearest'] = parameters.get('k_nearest', 0)
        from_ix = from_ix.to(dtype=torch.int32)
        to_ix = to_ix.to(dtype=torch.int32)

        # padded edges have weight 0 and probability 1
        with torch.no_grad():
            self.edge_adjacency_logits.data[:1].fill_(self.INF)
            self.edge_weight_logits.data[:1].fill_(self.NEG_INF)

        edge_adjacency_logits = F.embedding(
            torch.as_tensor(self.reorder_undirected_to_directed),
            self.edge_adjacency_logits, sparse=True
        )  # [num_edges, 1]
        edge_weight_logits = F.embedding(
            torch.as_tensor(self.reorder_undirected_to_directed),
            self.edge_weight_logits, sparse=True
        )  # [num_edges, 1]

        target_paths, nearest_paths = batch_dijkstra(
            self.directed_slices, self.directed_edge_indices,
            edge_adjacency_logits.data.numpy().flatten(),
            edge_weight_logits.data.numpy().flatten(),
            from_ix.data.numpy(), to_ix.data.numpy(),
            **parameters
        )

        target_paths = torch.as_tensor(target_paths, dtype=torch.int64)  # [batch_size, max_length]
        target_distances = F.softplus(self._get_logits(self.edge_weight_logits, target_paths)).sum(dim=(-1))
        logp_target_paths = -F.softplus(-self._get_logits(self.edge_adjacency_logits, target_paths)).sum(dim=(-1))

        # handle paths that are not found
        not_found_target = target_paths[..., 0] == 0
        if torch.any(not_found_target):
            is_not_loop = (from_ix[:, None] != to_ix.reshape(to_ix.shape[0], -1)).reshape(to_ix.shape)
            not_found_target = not_found_target & is_not_loop
        target_distances = torch.where(not_found_target, self._get_default_distance(), target_distances)

        if parameters['k_nearest'] != 0:
            nearest_paths = torch.as_tensor(np.copy(nearest_paths), dtype=torch.int64)
            nearest_distances = F.softplus(self._get_logits(self.edge_weight_logits, nearest_paths)).sum(dim=(-1))
            nearest_vertices = self.directed_edge_indices[nearest_paths[..., 0]]
        else:
            nearest_paths = nearest_distances = nearest_vertices = None

        return dict(
            target_paths=target_paths,
            target_distances=target_distances,
            logp_target_paths=logp_target_paths,
            found_target=~not_found_target,
            nearest_paths=nearest_paths,
            nearest_distances=nearest_distances,
            nearest_vertices=nearest_vertices,
        )

    def pruned(self, threshold=0.5):
        # Prunes graph edges by their adjacency probabilities
        # threshold: prunes all edges that have probability less than threshold
        # returns GraphEmbedding

        probs = torch.sigmoid(self.edge_adjacency_logits[1:, 0]).data.numpy()
        saved = probs >= threshold

        edges_from = self.edge_sources[1:]
        edges_to = self.edge_targets[1:]
        weights = F.softplus(self.edge_weight_logits[1:, 0]).data.numpy()

        return GraphEmbedding(edges_from[saved], edges_to[saved], weights=weights[saved],
                              probs=probs[saved])

    def report_model_size(self, threshold=0.5):
        # threshold: prunes all edges that have probability less than threshold
        # returns: byte size of graph, number of parameters and other statistics

        num_edges = check_numpy(
            torch.sigmoid(self.edge_adjacency_logits.flatten()[1:]) >= threshold
        ).astype('int64').sum()
        num_vertices = len(self.slices) + 1
        size_bytes = ((num_edges + num_vertices) * 32 + num_edges * 32) / 8
        return locals()

    def extra_repr(self): # class function
        edges_kept = np.sum(check_numpy(self.edge_adjacency_logits >= 0).astype('int64'))
        return "{} vertices, {} edges total, {} edges kept, {:.5} sparsity rate".format(
            self.num_vertices, self.num_edges, edges_kept, 1. - edges_kept / self.num_edges
        )
