# Branch detection functions using a thread pool
# Author: Jelmer Bot
# License: BSD 3 clause
import numpy as np
from joblib.parallel import delayed
from hdbscan.dist_metrics import DistanceMetric
from hdbscan._hdbscan_linkage import mst_linkage_core_vector
from hdbscan.branches import (
    extract_full_cluster_graph,
    extract_core_cluster_graph,
    compute_branch_linkage_from_graph,
    compute_branch_segmentation,
    update_labelling,
)

from ._hdbscan import hdbscan_mst_generic


def extract_cluster_points(cluster_labels, num_clusters):
    """Lists the data point indices in each cluster."""
    return [np.where(cluster_labels == l)[0] for l in range(num_clusters)]


def extract_cluster_msts(min_spanning_tree, cluster_labels, num_clusters):
    """List each cluster's minimum spanning tree from the global MST."""
    parent_label = cluster_labels[min_spanning_tree[:, 0].astype(np.intp)]
    child_label = cluster_labels[min_spanning_tree[:, 1].astype(np.intp)]
    return [
        min_spanning_tree[(parent_label == l) & (child_label == l)]
        for l in range(num_clusters)
    ]


def compute_branch_linkage(
    space_tree,  # None if run_generic
    reachability,  # None if not run_generic
    core_distances,
    neighbors,  # None if not run_core
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
        delayed(compute_branch_linkage_of_cluster)(
            space_tree,
            reachability[:, cluster_pts][cluster_pts, :] if run_generic else None,
            core_distances[cluster_pts],
            neighbors[cluster_pts, :] if run_core else None,
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
        for (cluster_mst, cluster_pts) in zip(cluster_spanning_trees, cluster_points)
    )
    if len(result):
        return tuple(zip(*result))
    return (), (), ()


def compute_branch_linkage_of_cluster(
    space_tree,  # None if run_generic
    reachability,  # None if not run_generic
    core_distances,
    neighbors,  # None if not run_core
    cluster_probabilities,
    cluster_spanning_tree,
    cluster_points,
    num_points,
    metric="minkowski",
    run_core=False,
    run_generic=False,
    run_override=False,
    **kwargs,
):
    """Computes depth-first-traversal distances in cluster_mst, extracts
    construct cluster approximation graph, and performs single linkage on the
    edge's centrality."""
    # Compute data-point centrality.
    if run_generic:
        centralities = 1 / np.max(reachability, axis=0)
    else:
        metric_fun = DistanceMetric.get_metric(metric, **kwargs)
        points = space_tree.data.base[cluster_points]
        centroid = np.average(points, weights=cluster_probabilities, axis=0)
        centralities = 1 / metric_fun.pairwise(centroid[None], points)[0, :]

    # within cluster ids
    cluster_ids = np.full(num_points, -1, dtype=np.double)
    cluster_ids[cluster_points] = np.arange(len(cluster_points), dtype=np.double)

    # Construct cluster approximation graph
    if run_core:
        if not run_override:
            cluster_spanning_tree[:, 0] = cluster_ids[
                cluster_spanning_tree[:, 0].astype(np.intp)
            ]
            cluster_spanning_tree[:, 1] = cluster_ids[
                cluster_spanning_tree[:, 1].astype(np.intp)
            ]
        edges = extract_core_cluster_graph(
            cluster_spanning_tree, core_distances, neighbors, cluster_ids
        )
    else:
        max_dist = cluster_spanning_tree.T[2].max()
        if run_generic:
            edges = _extract_full_cluster_graph_generic(reachability, max_dist)
        else:
            edges = extract_full_cluster_graph(
                space_tree, core_distances, cluster_points, cluster_ids, max_dist
            )

    # Add centrality to approximation graph
    return compute_branch_linkage_from_graph(
        cluster_points, centralities, edges, run_override
    )[1:]


def _extract_full_cluster_graph_generic(reachability, max_dist):
    """Create a cluster reachability graph."""
    # Allocate and fill output
    reachability = np.triu(reachability, 1)
    parents, children = np.where((reachability > 0.0) & (reachability <= max_dist))
    distances = reachability[parents, children]
    edges = np.column_stack((parents, children, np.zeros_like(distances), distances))
    return edges


def compute_msts_in_cluster_generic(
    reachability,
    cluster_points,
    thread_pool,
):
    """Computes MSTs in predefined clusters using Prims."""
    return thread_pool(
        delayed(hdbscan_mst_generic)(reachability[:, cluster_pts][cluster_pts, :])
        for cluster_pts in cluster_points
    )


def compute_msts_in_cluster_space_tree(
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
