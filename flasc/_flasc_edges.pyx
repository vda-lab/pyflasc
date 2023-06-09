# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Edgelist handling for FLASC
# Author: Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()


cpdef void _fill_edge_centrality(
    np.double_t[:, ::1] edges,
    np.double_t[::1] centralities
): # nogil:
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t num_edges = edges.shape[0]
    for idx in range(num_edges):
        edges[idx, 2] = max(
            centralities[<np.intp_t> edges[idx, 0]], 
            centralities[<np.intp_t> edges[idx, 1]]
        )


cpdef void _relabel_edges_with_data_ids(
    np.double_t[:, ::1] edges,
    np.intp_t[::1] cluster_points,
): # nogil:
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t num_edges = edges.shape[0]
    for idx in range(num_edges):
        edges[idx, 0] = <np.double_t> cluster_points[<np.intp_t> edges[idx, 0]]
        edges[idx, 1] = <np.double_t> cluster_points[<np.intp_t> edges[idx, 1]]


cpdef _extract_core_approximation_of_cluster(
    np.ndarray[np.double_t, ndim=2] cluster_spanning_tree,
    np.ndarray[np.double_t, ndim=1] core_distances,
    np.ndarray[np.intp_t, ndim=2] neighbours,
    np.ndarray[np.double_t, ndim=1] cluster_ids, # Full array, not just cluster
    bint run_override=False,
):
    """Create a graph connecting all points within each points core distance."""
    # Allocate output (won't be filled completely)
    cdef np.intp_t[:, ::1] neighbours_view = neighbours
    cdef np.double_t[:, ::1] cluster_spanning_tree_view = cluster_spanning_tree
    cdef np.intp_t num_points = neighbours_view.shape[0]
    cdef np.intp_t num_neighbours = neighbours_view.shape[1]
    cdef np.intp_t count = cluster_spanning_tree_view.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] edges = np.zeros((count + num_points * num_neighbours, 4), dtype=np.double)
    
    # Fill MST edges with within-cluster-ids
    cdef np.ndarray[np.double_t, ndim=1] mst_parents
    cdef np.ndarray[np.double_t, ndim=1] mst_children
    if run_override:
        mst_parents = cluster_spanning_tree[:, 0]
        mst_children = cluster_spanning_tree[:, 1]
    else:
        # (astype copy more effiecient than manual iteration)
        mst_parents = cluster_ids[cluster_spanning_tree[:, 0].astype(np.intp)]
        mst_children = cluster_ids[cluster_spanning_tree[:, 1].astype(np.intp)]
    np.minimum(mst_parents, mst_children, edges[:count, 0])
    np.maximum(mst_parents, mst_children, edges[:count, 1])
    
    # Fill neighbours with within-cluster-ids
    cdef np.ndarray[np.double_t, ndim=1] core_parent = np.repeat(
        np.arange(num_points, dtype=np.double), num_neighbours
    )
    cdef np.ndarray[np.double_t, ndim=1] core_children = cluster_ids[neighbours.flatten()]
    np.minimum(core_parent, core_children, edges[count:, 0])
    np.maximum(core_parent, core_children, edges[count:, 1])
    
    # Extract unique edges that stay within the cluster
    edges = np.unique(edges[edges[:, 0] > -1.0, :], axis=0)

    # Fill mutual reachabilities
    edges[:count, 3] = cluster_spanning_tree[:, 2]
    # (astype copy more effiecient than manual iteration)
    np.maximum(
        core_distances[edges[count:, 0].astype(np.intp)],
        core_distances[edges[count:, 1].astype(np.intp)],
        edges[count:, 3]
    )
    
    # Return output
    return edges
    

cpdef _extract_full_approximation_of_cluster_generic(
    np.double_t[:, ::1] reachability, np.double_t max_dist
):
    """Create a cluster reachability graph."""
    # Set lower triangle to zero
    reachability = np.triu(reachability, 1)
    
    # Extract the edge indices below the distance threshold
    cdef np.ndarray[np.intp_t, ndim=1] parents
    cdef np.ndarray[np.intp_t, ndim=1] children
    parents, children = np.where(
        (reachability.base > 0.0) & (reachability.base <= max_dist)
    )
    
    # Allocate and fill output
    cdef Py_ssize_t num_edges = len(parents)
    cdef np.ndarray[np.double_t, ndim=2] edges = np.zeros(
        (num_edges, 4), dtype=np.double
    )
    edges[:, 0] = parents
    edges[:, 1] = children
    edges[:, 3] = reachability.base[parents, children]
    return edges


cpdef _extract_full_approximation_of_cluster_space_tree(
    object space_tree,
    np.ndarray[np.double_t, ndim=1] core_distances,
    np.ndarray[np.intp_t, ndim=1] cluster_points,
    np.ndarray[np.double_t, ndim=1] cluster_ids,
    np.double_t max_dist
):
    # Query KDTree/BallTree for neighours within the distance
    cdef np.ndarray[object, ndim=1] children_map
    cdef np.ndarray[object, ndim=1] distances_map
    children_map, distances_map = space_tree.query_radius(
        space_tree.data.base[cluster_points], 
        r=max_dist + 1e-8, 
        return_distance=True
    )

    # Count number of returned edges 
    cdef np.intp_t i = 0
    cdef np.ndarray[np.intp_t, ndim=1] children
    cdef np.ndarray[np.intp_t, ndim=1] num_children = np.zeros(len(cluster_points), dtype=np.intp)
    for children in children_map:
        num_children[i] += len(children)
        i += 1
    cdef np.ndarray[np.double_t, ndim=1] full_parents = np.repeat(
        np.arange(len(cluster_points), dtype=np.double), num_children
    )
    cdef np.ndarray[np.double_t, ndim=1] full_children = cluster_ids[np.concatenate(children_map)]
    cdef np.ndarray[np.double_t, ndim=1] full_distances = np.concatenate(distances_map)

    # ALlocate output
    mask = (full_children != -1.0) & (full_parents < full_children) & (full_distances <= max_dist)
    cdef np.intp_t num_edge = mask.sum()
    cdef np.ndarray[np.double_t, ndim=2] edges = np.zeros((num_edge, 4), dtype=np.double)

    # Fill output
    edges[:, 0] = full_parents[mask]
    edges[:, 1] = full_children[mask]
    # (astype copies to int are more efficient than manual iteration here)
    np.maximum(np.maximum(
        core_distances[edges[:, 0].astype(np.intp)],
        core_distances[edges[:, 1].astype(np.intp)]
    ), full_distances[mask], edges[:, 3])

    # Return output
    return edges