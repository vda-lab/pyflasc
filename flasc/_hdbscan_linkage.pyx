# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Minimum spanning tree to single linkage implementation for hdbscan
# Original Authors: Leland McInnes, Steve Astels
# Adapted for fhdbscan by Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()


cpdef np.ndarray[np.double_t, ndim=2] mst_linkage_core(
    np.ndarray[np.double_t, ndim=2] distance_matrix
):
    """
    Computes MST from distance matrix.

    Compared to the original version, this one is updated to generate the same 
    output as Prims following https://github.com/scikit-learn-contrib/hdbscan/pull/315

    This is required because the post-processing step that corrects
    for equally distant points as a consequence of the mutual reachability 
    distance is removed.
    """
    cdef Py_ssize_t num_points = len(distance_matrix[:, 0])
    cdef np.ndarray[np.intp_t, ndim=1] current_labels = np.arange(num_points, dtype=np.intp)
    cdef np.ndarray[np.intp_t, ndim=1] current_sources = np.ones(num_points, dtype=np.intp)
    cdef np.ndarray[np.double_t, ndim=1] current_distances = np.infty * np.ones(num_points)
    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros((num_points - 1, 3))
    
    cdef np.intp_t i
    cdef np.intp_t current_node = 0
    cdef np.intp_t current_source
    cdef np.intp_t new_node
    cdef np.intp_t new_node_index

    cdef np.ndarray[np.double_t, ndim=1] left 
    cdef np.ndarray[np.double_t, ndim=1] right
    cdef np.ndarray label_filter, distance_mask

    for i in range(1, num_points):
        # Remove current node from list
        label_filter = current_labels != current_node
        current_labels = current_labels[label_filter]
        current_sources = current_sources[label_filter]
        # Find distances to other nodes from current node
        right = distance_matrix[current_node, current_labels]
        # Find shortest distances to other nodes
        left = current_distances[label_filter]
        # Update shortest distances where current node is closest
        distance_mask = left < right
        current_distances = np.where(distance_mask, left, right)
        # Update source nodes where current node is closest
        current_sources = np.where(distance_mask, current_sources, current_node)
        # Find closest point to current node
        new_node_index = np.argmin(current_distances)
        new_node = current_labels[new_node_index]
        current_source = current_sources[new_node_index]
        # Create edges between them
        result[i - 1, 0] = <double> current_source
        result[i - 1, 1] = <double> new_node
        result[i - 1, 2] = current_distances[new_node_index]
        # Continue with the new node
        current_node = new_node
    return result