# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Approximation graph traversal logic
# Author: Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.deque cimport deque
from libcpp.vector cimport vector


cpdef np.ndarray[np.double_t, ndim=1] _compute_cluster_centralities(
    np.double_t[:, ::1] cluster_core_graph,
    np.intp_t cluster_root,
    np.intp_t num_points
):
    # Convert MST to {parent: [children], child: [parents]} format for traversal.
    cdef Py_ssize_t idx = 0
    cdef np.double_t parent = 0
    cdef np.double_t child = 0
    cdef Py_ssize_t num_edges = cluster_core_graph.shape[0]
    cdef vector[vector[np.double_t]] network = vector[vector[np.double_t]](num_points)
    #with nogil:
    for idx in range(num_edges):
        parent = cluster_core_graph[idx, 0]
        child = cluster_core_graph[idx, 1]
        network[<np.intp_t> parent].push_back(child)
        network[<np.intp_t> child].push_back(parent)

    # Allocate output
    cdef np.ndarray[np.double_t, ndim=1] depths = np.inf * np.ones(num_points, dtype=np.double)
    cdef np.double_t[::1] depths_view = depths

    # Traversal variables
    cdef np.double_t grand_child = 0 
    cdef np.double_t depth=0
    cdef np.double_t max_depth = 0
    cdef pair[np.double_t, np.double_t] edge
    cdef pair[pair[np.double_t, np.double_t], np.double_t] item
    cdef deque[pair[pair[np.double_t, np.double_t], np.double_t]] queue
    cdef vector[bool] flags = vector[bool](num_points)
    
    # Traverse the network
    # with nogil:
    # Queue the root's children
    edge.first = <np.double_t> cluster_root
    depths_view[cluster_root] = 0.0
    flags[cluster_root] = True
    for child in network[cluster_root]:
        edge.second = child
        item.first = edge
        item.second = 1.0
        queue.push_back(item)
        flags[<np.intp_t> child] = True
    
    # Traverse down the children
    while not queue.empty():
        # Extract values
        item = queue.front()
        queue.pop_front()
        parent = item.first.first
        child = item.first.second
        depth = item.second

        # Fill in the depth value, keep track of max
        depths_view[<np.intp_t> child] = depth
        max_depth = max(depth, max_depth)

        # Enqueue grand-children
        item.second += 1.0
        edge.first = child
        for grand_child in network[<np.intp_t> child]:
            if flags[<np.intp_t> grand_child]:
                continue
            edge.second = grand_child
            item.first = edge
            queue.push_back(item)
            flags[<np.intp_t> grand_child] = True
    return max_depth - depths