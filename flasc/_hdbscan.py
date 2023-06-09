# Adapted HDBSCAN internals
# Original Authors: Leland McInnes <leland.mcinnes@gmail.com>
#                   Steve Astels <sastels@gmail.com>
#                   John Healy <jchealy@gmail.com>
# Adapted for FLASC by Jelmer Bot
# License: BSD 3 clause

import numpy as np
from warnings import warn
from joblib.parallel import delayed
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KDTree, BallTree

from hdbscan.dist_metrics import DistanceMetric
from hdbscan._hdbscan_linkage import mst_linkage_core_vector, label
from hdbscan._hdbscan_tree import compute_stability, condense_tree
from ._hdbscan_linkage import mst_linkage_core
from ._hdbscan_boruvka import KDTreeBoruvkaAlgorithm, BallTreeBoruvkaAlgorithm
from ._hdbscan_tree import get_clusters

# ---- Generic
def _hdbscan_generic_linkage(
    X, min_samples=5, alpha=1.0, metric="minkowski", **kwargs
):
    """Computes HDBSCAN single linkage from input."""
    # Compute mutual reachability
    reachability, core_distances, neighbours = _hdbscan_generic_reachability(
        X,
        min_samples=min_samples,
        alpha=alpha,
        metric=metric,
        **kwargs,
    )

    # Compute mst from reachability
    min_spanning_tree = mst_linkage_core(reachability)
    if np.isinf(min_spanning_tree.T[2]).any():
        warn(
            "The minimum spanning tree contains edge weights with value "
            "infinity. Potentially, you are missing too many distances "
            "in the initial distance matrix for the given neighbourhood "
            "size.",
            UserWarning,
        )

    # Compute single linkage hierarchy
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    single_linkage_tree = label(min_spanning_tree)
    return (
        single_linkage_tree,
        min_spanning_tree,
        reachability,
        core_distances,
        neighbours,
    )


def _hdbscan_generic_reachability(
    X, min_samples=5, alpha=1.0, metric="minkowski", p=2, **kwargs
):
    """Computes full reachability matrix."""
    if metric == "minkowski":
        distance_matrix = pairwise_distances(X, metric=metric, p=p)
    elif metric == "arccos":
        distance_matrix = pairwise_distances(X, metric="cosine", **kwargs)
    elif metric == "precomputed":
        distance_matrix = X.copy()
    else:
        distance_matrix = pairwise_distances(X, metric=metric, **kwargs)

    return _hdbscan_generic_mutual_reachability(
        distance_matrix,
        min_points=min_samples,
        alpha=alpha,
    )


def _hdbscan_generic_mutual_reachability(
    distance_matrix,
    min_points=5,
    alpha=1.0,
):
    """Compute the weighted adjacency matrix of the mutual reachability
    graph of a distance matrix.
    """
    # Find all min_points closest neighbours
    np.fill_diagonal(distance_matrix, np.nan)
    neighbours = np.argpartition(distance_matrix, min_points - 1)[
        :, :min_points
    ]

    # Extract the core distance
    core_distances = np.take_along_axis(
        distance_matrix, neighbours[:, -1][None], axis=0
    )[0]

    # Update with alpha if applied
    if alpha != 1.0:
        distance_matrix = distance_matrix / alpha

    # Apply the core distance to compute the mutual reachability
    stage1 = np.where(
        core_distances > distance_matrix, core_distances, distance_matrix
    )
    result = np.where(core_distances > stage1.T, core_distances.T, stage1.T).T
    np.fill_diagonal(result, core_distances)

    return result, core_distances, neighbours


# --- Space tree
def _hdbscan_space_tree_linkage(
    space_tree,
    min_samples=5,
    metric="minkowski",
    alpha=1.0,
    algorithm="best",
    leaf_size=40,
    approx_min_span_tree=True,
    thread_pool=None,
    **kwargs
):
    """Computes HDBSCAN single linkage from space tree."""
    (min_spanning_tree, core_distances, neighbours) = _hdbscan_space_tree_mst(
        space_tree,
        min_samples=min_samples,
        alpha=alpha,
        metric=metric,
        algorithm=algorithm,
        leaf_size=leaf_size,
        approx_min_span_tree=approx_min_span_tree,
        thread_pool=thread_pool,
        **kwargs,
    )
    min_spanning_tree = min_spanning_tree[np.argsort(min_spanning_tree.T[2]), :]
    single_linkage_tree = label(min_spanning_tree)
    return (single_linkage_tree, min_spanning_tree, core_distances, neighbours)


def _hdbscan_space_tree_mst(
    space_tree,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    algorithm="best",
    leaf_size=40,
    approx_min_span_tree=True,
    thread_pool=None,
    **kwargs
):
    # Select which MST algorithm to use.
    if algorithm.startswith("prims"):
        Algorithm = _hdbscan_space_tree_mst_prims
    elif algorithm.startswith("boruvka"):
        Algorithm = _hdbscan_space_tree_mst_boruvka
    elif len(space_tree.data[0, :]) > 60:
        Algorithm = _hdbscan_space_tree_mst_prims
    else:
        Algorithm = _hdbscan_space_tree_mst_boruvka

    return Algorithm(
        space_tree,
        min_samples=min_samples,
        alpha=alpha,
        metric=metric,
        leaf_size=leaf_size,
        approx_min_span_tree=approx_min_span_tree,
        thread_pool=thread_pool,
        **kwargs,
    )


def _hdbscan_space_tree_mst_prims(
    space_tree,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    leaf_size=None,
    approx_min_span_tree=None,
    thread_pool=None,
    **kwargs
):
    """Computes MST from space tree using prims algorithm."""
    core_distances, neighbours = _hdbscan_space_tree_core_dists_prims(
        space_tree,
        min_samples=min_samples,
        thread_pool=thread_pool,
    )
    dist_metric = DistanceMetric.get_metric(metric, **kwargs)
    mst = mst_linkage_core_vector(
        space_tree.data.base, core_distances, dist_metric, alpha
    )
    return mst, core_distances, neighbours


def _hdbscan_space_tree_core_dists_prims(
    space_tree, min_samples=5, thread_pool=None
):
    """Finds the min_samples closest points and the core distance."""
    if thread_pool.n_jobs == 1:
        core_distances, neighbours = space_tree.query(
            space_tree.data.base,
            k=min_samples + 1,
            dualtree=True,
            breadth_first=True,
        )
    else:
        datasets = []
        num_jobs = min(thread_pool.n_jobs, space_tree.data.shape[0])
        split_cnt = max(space_tree.data.shape[0] // thread_pool.n_jobs, 1)
        for i in range(num_jobs):
            if i == num_jobs - 1:
                datasets.append(space_tree.data.base[i * split_cnt :])
            else:
                datasets.append(
                    space_tree.data.base[i * split_cnt : (i + 1) * split_cnt]
                )

        knn_data = thread_pool(
            delayed(space_tree.query)(
                points, k=min_samples + 1, dualtree=True, breadth_first=True
            )
            for points in datasets
        )

        core_distances = np.vstack([x[0] for x in knn_data])
        neighbours = np.vstack([x[1] for x in knn_data])

    core_distances = core_distances[:, -1].copy(order="C")
    neighbours = neighbours[:, 1:]
    return core_distances, neighbours


def _hdbscan_space_tree_mst_boruvka(
    space_tree,
    min_samples=5,
    alpha=1.0,
    metric="minkowski",
    leaf_size=40,
    approx_min_span_tree=True,
    thread_pool=None,
    **kwargs
):
    if isinstance(space_tree, KDTree):
        Algorithm = KDTreeBoruvkaAlgorithm
    else:
        Algorithm = BallTreeBoruvkaAlgorithm

    alg = Algorithm(
        space_tree,
        min_samples=min_samples,
        metric=metric,
        alpha=alpha,
        leaf_size=max(leaf_size, 3) // 3,
        approx_min_span_tree=approx_min_span_tree,
        thread_pool=thread_pool,
        **kwargs,
    )
    min_spanning_tree = alg.spanning_tree()
    if isinstance(space_tree, KDTree):
        core_distances = np.sqrt(alg.core_distance_arr)
    else:
        core_distances = alg.core_distance_arr  # copy()
    neighbours = alg.neighbour_arr  # copy()
    return min_spanning_tree, core_distances, neighbours


def _hdbscan_space_tree(
    X, metric="minkowski", algorithm="best", leaf_size=40, **kwargs
):
    """Computes a space tree from the input."""
    # Select which tree type to build
    if algorithm.endswith("kdtree"):
        TreeType = KDTree
    elif algorithm.endswith("balltree"):
        TreeType = BallTree
    elif metric in KDTree.valid_metrics:
        TreeType = KDTree
    else:
        TreeType = BallTree

    return TreeType(X, metric=metric, leaf_size=leaf_size, **kwargs)


# --- Shared


def _hdbscan_extract_clusters(
    single_linkage_tree,
    min_cluster_size=5,
    cluster_selection_method="eom",
    allow_single_cluster=False,
    cluster_selection_epsilon=0.0,
    max_cluster_size=0,
):
    """Extracts clusters from single linkage hierarchy."""
    condensed_tree = condense_tree(single_linkage_tree, min_cluster_size)
    stability_dict = compute_stability(condensed_tree)
    (labels, probabilities, stabilities) = get_clusters(
        condensed_tree,
        stability_dict,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
    )
    return labels, probabilities, stabilities, condensed_tree
