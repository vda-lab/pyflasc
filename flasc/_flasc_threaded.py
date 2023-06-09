# Branch detection functions using a thread pool
# Author: Jelmer Bot
# License: BSD 3 clause
from joblib.parallel import delayed
from hdbscan._hdbscan_linkage import mst_linkage_core_vector
from hdbscan.dist_metrics import DistanceMetric

from ._hdbscan_linkage import mst_linkage_core
from ._flasc_branches import (
    _compute_branch_linkage_of_cluster,
    _compute_branch_segmentation_of_cluster,
)


def _compute_branch_linkage(
    space_tree,  # None if run_generic
    reachability,  # None if not run_generic
    core_distances,
    neighbours,  # None if not run_core
    cluster_probabilities,
    cluster_spanning_trees,
    cluster_points,
    thread_pool,
    metric="minkowski",
    run_core=False,
    run_generic=False,
    run_override=False,
    **kwargs,
):
    """Computes point-wise depths and single linkage hierarchy for each cluster
    using the given thread pool."""
    result = thread_pool(
        delayed(_compute_branch_linkage_of_cluster)(
            space_tree,
            reachability[:, cluster_pts][cluster_pts, :] if run_generic else None,
            core_distances[cluster_pts],
            neighbours[cluster_pts, :] if run_core else None,
            cluster_probabilities[cluster_pts],
            cluster_mst,
            cluster_pts,
            len(cluster_probabilities),
            metric=metric,
            run_core=run_core,
            run_generic=run_generic,
            run_override=run_override,
            **kwargs,
        )
        for (cluster_mst, cluster_pts) in zip(
            cluster_spanning_trees, cluster_points
        )
    )
    if len(result):
        return tuple(zip(*result))
    return (), (), ()


def _compute_branch_segmentation(
    branch_linkage_trees,
    thread_pool,
    min_branch_size=5,
    allow_single_branch=False,
    branch_selection_method="eom",
    branch_selection_persistence=0.0,
    max_branch_size=0,
):
    """Extracts branches from the linkage hierarchies."""
    results = thread_pool(
        delayed(_compute_branch_segmentation_of_cluster)(
            branch_linkage_tree,
            min_branch_size=min_branch_size,
            allow_single_branch=allow_single_branch,
            branch_selection_method=branch_selection_method,
            branch_selection_persistence=branch_selection_persistence,
            max_branch_size=max_branch_size,
        )
        for branch_linkage_tree in branch_linkage_trees
    )
    if len(results):
        return tuple(zip(*results))
    return (), (), (), ()


def _compute_msts_in_cluster_generic(
    reachability,
    cluster_points,
    thread_pool,
):
    """Computes MSTs in predefined clusters using Prims."""
    return thread_pool(
        delayed(mst_linkage_core)(reachability[:, cluster_pts][cluster_pts, :])
        for cluster_pts in cluster_points
    )


def _compute_msts_in_cluster_space_tree(
    space_tree,
    core_distances,
    cluster_points,
    thread_pool,
    metric="minkowski",
    alpha=1.0,
    **kwargs,
):
    """Computes MSTs in predefined clusters using Prims."""
    dist_metric = DistanceMetric.get_metric(metric, **kwargs)
    return thread_pool(
        delayed(mst_linkage_core_vector)(
            space_tree.data.base[cluster_pts, :],
            core_distances[cluster_pts],
            dist_metric,
            alpha,
        )
        for cluster_pts in cluster_points
    )
