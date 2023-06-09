# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Minimum spanning tree to single linkage implementation for fhdbscan
# Original Authors: Leland McInnes, Steve Astels
# Adapted for FLASC by Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()


cpdef np.ndarray[np.double_t, ndim=2] label(
    np.double_t[:, ::1] edges, 
    np.intp_t num_points
):
    """Convert an edge list into single linkage hierarchy."""
    # ALlocate output and working structure
    cdef np.intp_t N = edges.shape[0]
    cdef np.ndarray[np.double_t, ndim=2] result = np.zeros((N, 4))
    cdef np.double_t[:, ::1] result_view = result
    cdef UnionFind U = UnionFind(N, num_points)
    
    cdef np.intp_t a, aa, b, bb, idx, size
    cdef np.intp_t cnt = 0
    
    # with nogil:
    for idx in range(edges.shape[0]):
        a = <np.intp_t> edges[idx, 0]
        b = <np.intp_t> edges[idx, 1]
        aa = U.fast_find(a)
        bb = U.fast_find(b)

        if aa == bb:
            continue
        size = U.union(aa, bb)
        result_view[cnt, 0] = <np.double_t> aa
        result_view[cnt, 1] = <np.double_t> bb
        result_view[cnt, 2] = edges[idx, 2]
        result_view[cnt, 3] = <np.double_t> size
        cnt = cnt + 1

    return result[:cnt,:]


cdef class UnionFind (object):
    cdef np.intp_t next_label
    cdef np.ndarray parent_arr
    cdef np.ndarray size_arr
    cdef np.intp_t[::1] parent
    cdef np.intp_t[::1] size

    def __init__(self, N, P):
        self.next_label = P
        
        self.parent_arr = np.full(P + N, -1, dtype=np.intp)
        self.size_arr = np.ones(P + N, dtype=np.intp)

        self.parent = self.parent_arr
        self.size = self.size_arr

    cdef np.intp_t union(self, np.intp_t m, np.intp_t n): # nogil:
        cdef np.intp_t out = self.size[m] + self.size[n]
        self.parent[m] = self.next_label
        self.parent[n] = self.next_label
        self.size[self.next_label] = out
        self.next_label += 1
        return out

    cdef np.intp_t fast_find(self, np.intp_t n): # nogil:
        cdef np.intp_t p = n
        while self.parent[n] != -1:
            n = self.parent[n]
        # label up to the root
        while self.parent[p] != n:
            p, self.parent[p] = self.parent[p], n
        return n