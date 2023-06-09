# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Internal Branch detection functions
# Author: Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()

from hdbscan._hdbscan_tree import compute_stability, condense_tree

from ._hdbscan_dist_metrics import DistanceMetric
from ._hdbscan_dist_metrics cimport DistanceMetric
from ._flasc_linkage import label
from ._flasc_tree import get_clusters
from ._flasc_edges import (
    _fill_edge_centrality,
    _relabel_edges_with_data_ids,
    _extract_core_approximation_of_cluster,
    _extract_full_approximation_of_cluster_generic,
    _extract_full_approximation_of_cluster_space_tree,
)


def _extract_cluster_points(
    np.ndarray[np.intp_t, ndim=1] cluster_labels, 
    Py_ssize_t num_clusters
):
    """Lists the data point indices in each cluster."""
    cdef np.intp_t l 
    return [
        np.where(cluster_labels == l)[0] for l in range(num_clusters)
    ]


def _extract_cluster_msts(
    np.ndarray[np.double_t, ndim=2] min_spanning_tree,
    np.ndarray[np.intp_t, ndim=1] cluster_labels, 
    np.intp_t num_clusters
):
    """List each cluster's minimum spanning tree from the global MST."""
    cdef np.intp_t l, idx 
    cdef np.double_t[::1] edge
    cdef np.double_t[:, ::1] tree_view = min_spanning_tree
    cdef np.ndarray[np.intp_t, ndim=1] parent_label = np.empty(tree_view.shape[0], dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] child_label = np.empty(tree_view.shape[0], dtype=np.intp)
    for idx, edge in enumerate(tree_view):
        parent_label[idx] = cluster_labels[<np.intp_t> edge[0]]
        child_label[idx] = cluster_labels[<np.intp_t> edge[1]]

    return [
        min_spanning_tree[(parent_label == l) & (child_label == l)]
        for l in range(num_clusters)
    ]


def _compute_branch_linkage_of_cluster(
    space_tree, # None if run_generic
    reachability, # None if not run_generic
    np.ndarray[np.double_t, ndim=1] core_distances,
    np.ndarray[np.intp_t, ndim=2] neighbours, # None if not run_core
    np.ndarray[np.double_t, ndim=1] cluster_probabilities,
    np.ndarray[np.double_t, ndim=2] cluster_spanning_tree,
    np.ndarray[np.intp_t, ndim=1] cluster_points, 
    np.intp_t num_points,
    metric='minkowski',
    bint run_core=False,
    bint run_generic=False,
    bint run_override=False,
    **kwargs
):
    """Computes depth-first-traversal distances in cluster_mst, extracts 
    construct cluster approximation graph, and performs single linkage on the 
    edge's centrality."""
    # Compute data-point centrality.
    cdef DistanceMetric metric_fun
    cdef np.double_t[:, ::1] points
    cdef np.double_t[::1] centroid
    cdef np.ndarray[np.double_t, ndim=1] centralities
    if run_generic:
        centralities = np.mean(reachability, axis=0)
    else:
        metric_fun = DistanceMetric.get_metric(metric, **kwargs)
        points = space_tree.data.base[cluster_points]
        centroid = np.average(points, weights=cluster_probabilities, axis=0)
        centralities = metric_fun.pairwise(centroid[None], points)[0, :]
    centralities = centralities.max() - centralities

    # within cluster ids
    cdef np.ndarray[np.double_t, ndim=1] cluster_ids = np.full(num_points, -1, dtype=np.double)
    cluster_ids[cluster_points] = np.arange(len(cluster_points), dtype=np.double)

    # Construct cluster approximation graph
    cdef np.double_t max_dist
    cdef np.ndarray[np.double_t, ndim=2] edges    
    if run_core:
        edges = _extract_core_approximation_of_cluster(
            cluster_spanning_tree, core_distances, neighbours, cluster_ids, 
            run_override
        )
    else:
        max_dist = cluster_spanning_tree.T[2].max()
        if run_generic:
            edges = _extract_full_approximation_of_cluster_generic(
                reachability, max_dist
            )
        else:
            edges = _extract_full_approximation_of_cluster_space_tree(
                space_tree, core_distances, cluster_points, cluster_ids, max_dist
            )
        
    # Add centrality to approximation graph
    _fill_edge_centrality(edges, centralities)
    
    # Sort and compute single linkage
    edges = edges[np.argsort(edges.T[2]), :]
    cdef np.ndarray[np.double_t, ndim=2] linkage_tree = label(edges, len(cluster_points))

    # Re-label edges with data ids
    _relabel_edges_with_data_ids(edges, cluster_points)

    # Return values
    return centralities, linkage_tree, edges
    

def _compute_branch_segmentation_of_cluster(
    np.double_t[:, ::1] single_linkage_tree,
    np.intp_t min_branch_size=5,
    bint allow_single_branch=False,
    str branch_selection_method='eom',
    np.double_t branch_selection_persistence=0.0,
    np.intp_t max_branch_size=0
):
    """Simplifies the linkage tree"""
    cdef np.ndarray condensed_tree = condense_tree(
        single_linkage_tree.base, min_branch_size
    )
    cdef dict stability = compute_stability(condensed_tree)
    (labels, probabilities, persistences) = get_clusters(
        condensed_tree, stability,
        allow_single_branch=allow_single_branch,
        branch_selection_method=branch_selection_method,
        branch_selection_persistence=branch_selection_persistence,
        max_branch_size=max_branch_size
    )
    # Reset noise labels to 0-cluster
    labels[labels < 0] = len(persistences)
    return (labels, probabilities, persistences, condensed_tree)


def _update_labelling(
    np.ndarray[np.intp_t, ndim=1] cluster_labels,
    np.ndarray[np.double_t, ndim=1] cluster_probabilities,
    list cluster_points_,
    tuple cluster_depths_,
    tuple branch_labels_,
    tuple branch_probabilities_,
    tuple branch_persistences_,
    bint label_sides_as_branches=False,
):
    # Allocate output
    cdef Py_ssize_t num_points = len(cluster_labels)
    cdef np.ndarray[np.intp_t, ndim=1] labels = -1 * np.ones(num_points, dtype=np.intp)
    cdef np.ndarray[np.double_t, ndim=1] probabilities = cluster_probabilities.copy()
    cdef np.ndarray[np.intp_t, ndim=1] branch_labels = np.zeros(num_points, dtype=np.intp)
    cdef np.ndarray[np.double_t, ndim=1] branch_probabilities = np.ones(num_points, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] branch_depths = np.zeros(num_points, dtype=np.double)
    
    # Compute the labels and probabilities
    cdef Py_ssize_t num_branches = 0
    cdef np.intp_t running_id = 0, cid = 0
    cdef np.ndarray[np.intp_t, ndim=1] _points, _labels
    cdef np.ndarray[np.double_t, ndim=1] _probs, _pers, _depths
    for _points, _depths, _labels, _probs, _pers in zip(
        cluster_points_, 
        cluster_depths_, 
        branch_labels_, 
        branch_probabilities_,
        branch_persistences_
    ):
        num_branches = len(_pers)
        branch_depths[_points] = _depths
        if num_branches <= (1 if label_sides_as_branches else 2):
            labels[_points] = running_id
            running_id += 1
        else:
            labels[_points] = _labels + running_id
            branch_labels[_points] = _labels
            branch_probabilities[_points] = _probs
            probabilities[_points] += _probs
            probabilities[_points] /= 2
            running_id += num_branches + 1
    
    # Reorder other parts
    return (
        labels, 
        probabilities, 
        branch_labels, 
        branch_probabilities,
        branch_depths
    )