# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
# Condensing Tree handling for hdbscan
# Original Author: Leland McInnes
# Adapted for fhdbscan by Jelmer Bot
# License: BSD 3 clause
import numpy as np
cimport numpy as np
np.import_array()

cdef np.double_t INFTY = np.inf

from hdbscan._hdbscan_tree import (
    get_stability_scores,
    get_cluster_tree_leaves,
    epsilon_search
)


cdef list bfs_from_cluster_tree(np.ndarray tree, np.intp_t bfs_root):
    cdef list result
    cdef np.ndarray[np.intp_t, ndim=1] to_process

    result = []
    to_process = np.array([bfs_root], dtype=np.intp)

    while len(to_process) > 0:
        result.extend(to_process.tolist())
        to_process = tree['child'][np.in1d(tree['parent'], to_process)]

    return result


cdef max_lambdas(np.ndarray tree):
    cdef np.ndarray sorted_parent_data
    cdef np.ndarray[np.intp_t, ndim=1] sorted_parents
    cdef np.ndarray[np.double_t, ndim=1] sorted_lambdas
    cdef np.intp_t[:] sorted_parents_view
    cdef np.double_t[:] sorted_lambdas_view

    cdef np.intp_t parent
    cdef np.intp_t current_parent
    cdef np.double_t lambda_
    cdef np.double_t max_lambda

    cdef np.ndarray[np.double_t, ndim=1] deaths_arr
    cdef np.double_t[::1] deaths

    cdef np.intp_t largest_parent = tree['parent'].max()

    sorted_parent_data = np.sort(tree[['parent', 'lambda_val']], axis=0)
    deaths_arr = np.zeros(largest_parent + 1, dtype=np.double)
    deaths = deaths_arr
    sorted_parents = sorted_parent_data['parent']
    sorted_lambdas = sorted_parent_data['lambda_val']
    sorted_parents_view = sorted_parents
    sorted_lambdas_view = sorted_lambdas

    current_parent = -1
    max_lambda = 0

    for parent, lambda_ in zip(sorted_parents_view, sorted_lambdas_view):
        if parent == current_parent:
            max_lambda = max(max_lambda, lambda_)
        elif current_parent != -1:
            deaths[current_parent] = max_lambda
            current_parent = parent
            max_lambda = lambda_
        else:
            # Initialize
            current_parent = parent
            max_lambda = lambda_
    
    deaths[current_parent] = max_lambda # value for last parent

    return deaths_arr


cdef class TreeUnionFind (object):
    cdef np.ndarray _data_arr
    cdef np.intp_t[:, ::1] _data
    cdef np.ndarray is_component

    def __init__(self, size):
        self._data_arr = np.zeros((size, 2), dtype=np.intp)
        self._data_arr.T[0] = np.arange(size)
        self._data = self._data_arr
        self.is_component = np.ones(size, dtype=bool)

    cdef union_(self, np.intp_t x, np.intp_t y):
        cdef np.intp_t x_root = self.find(x)
        cdef np.intp_t y_root = self.find(y)

        if self._data[x_root, 1] < self._data[y_root, 1]:
            self._data[x_root, 0] = y_root
        elif self._data[x_root, 1] > self._data[y_root, 1]:
            self._data[y_root, 0] = x_root
        else:
            self._data[y_root, 0] = x_root
            self._data[x_root, 1] += 1

        return

    cdef find(self, np.intp_t x):
        if self._data[x, 0] != x:
            self._data[x, 0] = self.find(self._data[x, 0])
            self.is_component[x] = False
        return self._data[x, 0]

    cdef np.ndarray[np.intp_t, ndim=1] components(self):
        return self.is_component.nonzero()[0]


cdef np.ndarray[np.intp_t, ndim=1] do_labelling(
    np.ndarray tree,
    set clusters,
    dict cluster_label_map,
    np.intp_t allow_single_cluster,
    np.double_t cluster_selection_epsilon
):
    cdef np.intp_t root_cluster
    cdef np.ndarray[np.intp_t, ndim=1] result_arr
    cdef np.ndarray[np.intp_t, ndim=1] parent_array
    cdef np.ndarray[np.intp_t, ndim=1] child_array
    cdef np.ndarray[np.double_t, ndim=1] lambda_array
    cdef np.intp_t[::1] result
    cdef TreeUnionFind union_find
    cdef np.intp_t parent
    cdef np.intp_t child
    cdef np.intp_t n
    cdef np.intp_t cluster

    child_array = tree['child']
    parent_array = tree['parent']
    lambda_array = tree['lambda_val']

    root_cluster = parent_array.min()
    result_arr = np.empty(root_cluster, dtype=np.intp)
    result = result_arr

    union_find = TreeUnionFind(parent_array.max() + 1)

    for n in range(len(child_array)):
        child = child_array[n]
        parent = parent_array[n]
        if child not in clusters:
            union_find.union_(parent, child)

    for n in range(root_cluster):
        cluster = union_find.find(n)
        if cluster < root_cluster:
            result[n] = -1
        elif cluster == root_cluster:
            if len(clusters) == 1 and allow_single_cluster:
                # ALl points get the cluster label except when epsilon is specified
                if cluster_selection_epsilon != 0.0:
                    if tree['lambda_val'][tree['child'] == n] >= 1 / cluster_selection_epsilon :
                        result[n] = cluster_label_map[cluster]
                    else:
                        result[n] = -1
                else:
                    result[n] = cluster_label_map[cluster]
            else:
                result[n] = -1
        else:
            result[n] = cluster_label_map[cluster]

    return result_arr


cpdef get_probabilities(np.ndarray tree, dict cluster_map, np.ndarray labels):
    cdef np.ndarray[np.double_t, ndim=1] result
    cdef np.ndarray[np.double_t, ndim=1] deaths
    cdef np.ndarray[np.double_t, ndim=1] lambda_array
    cdef np.ndarray[np.intp_t, ndim=1] child_array
    cdef np.ndarray[np.intp_t, ndim=1] parent_array
    cdef np.intp_t root_cluster
    cdef np.intp_t n
    cdef np.intp_t point
    cdef np.intp_t cluster_num
    cdef np.intp_t cluster
    cdef np.double_t max_lambda
    cdef np.double_t lambda_

    child_array = tree['child']
    parent_array = tree['parent']
    lambda_array = tree['lambda_val']

    result = np.zeros(len(labels))
    deaths = max_lambdas(tree)
    root_cluster = parent_array.min()

    for n in range(len(child_array)):
        point = child_array[n]
        if point >= root_cluster:
            continue

        cluster_num = labels[point]

        if cluster_num == -1:
            continue

        cluster = cluster_map[cluster_num]
        max_lambda = deaths[cluster]
        if max_lambda == 0.0 or not np.isfinite(lambda_array[n]):
            result[point] = 1.0
        else:
            lambda_ = min(lambda_array[n], max_lambda)
            result[point] = lambda_ / max_lambda

    return result


cpdef tuple get_clusters(np.ndarray tree, dict stability,
                         cluster_selection_method='eom',
                         allow_single_cluster=False,
                         cluster_selection_epsilon=0.0,
                         max_cluster_size=0):
    """Given a tree and stability dict, produce the cluster labels
    (and probabilities) for a flat clustering based on the chosen
    cluster selection method.
    Parameters
    ----------
    tree : numpy recarray
        The condensed tree to extract flat clusters from
    stability : dict
        A dictionary mapping cluster_ids to stability values
    cluster_selection_method : string, optional (default 'eom')
        The method of selecting clusters. The default is the
        Excess of Mass algorithm specified by 'eom'. The alternate
        option is 'leaf'.
    allow_single_cluster : boolean, optional (default False)
        Whether to allow a single cluster to be selected by the
        Excess of Mass algorithm.
    cluster_selection_epsilon: float, optional (default 0.0)
        A distance threshold for cluster splits.
        
    max_cluster_size: int, optional (default 0)
        The maximum size for clusters located by the EOM clusterer. Can
        be overridden by the cluster_selection_epsilon parameter in
        rare cases.
    Returns
    -------
    labels : ndarray (n_samples,)
        An integer array of cluster labels, with -1 denoting noise.
    probabilities : ndarray (n_samples,)
        The cluster membership strength of each sample.
    stabilities : ndarray (n_clusters,)
        The cluster coherence strengths of each cluster.
    """
    cdef list node_list
    cdef np.ndarray cluster_tree
    cdef np.ndarray child_selection
    cdef dict is_cluster
    cdef dict cluster_sizes
    cdef float subtree_stability
    cdef np.intp_t node
    cdef np.intp_t sub_node
    cdef np.intp_t cluster
    cdef np.intp_t num_points
    cdef np.ndarray labels
    cdef np.double_t max_lambda

    # Assume clusters are ordered by numeric id equivalent to
    # a topological sort of the tree; This is valid given the
    # current implementation above, so don't change that ... or
    # if you do, change this accordingly!
    if allow_single_cluster:
        node_list = sorted(stability.keys(), reverse=True)
    else:
        node_list = sorted(stability.keys(), reverse=True)[:-1]
        # (exclude root)

    cluster_tree = tree[tree['child_size'] > 1]
    is_cluster = {cluster: True for cluster in node_list}
    num_points = np.max(tree[tree['child_size'] == 1]['child']) + 1
    max_lambda = np.max(tree['lambda_val'])

    if max_cluster_size <= 0:
        max_cluster_size = num_points + 1  # Set to a value that will never be triggered
    cluster_sizes = {child: child_size for child, child_size
                 in zip(cluster_tree['child'], cluster_tree['child_size'])}
    if allow_single_cluster:
        # Compute cluster size for the root node
        cluster_sizes[node_list[-1]] = np.sum(
            cluster_tree[cluster_tree['parent'] == node_list[-1]]['child_size'])

    if cluster_selection_method == 'eom':
        for node in node_list:
            child_selection = (cluster_tree['parent'] == node)
            subtree_stability = np.sum([
                stability[child] for
                child in cluster_tree['child'][child_selection]])
            if subtree_stability > stability[node] or cluster_sizes[node] > max_cluster_size:
                is_cluster[node] = False
                stability[node] = subtree_stability
            else:
                for sub_node in bfs_from_cluster_tree(cluster_tree, node):
                    if sub_node != node:
                        is_cluster[sub_node] = False

        if cluster_selection_epsilon != 0.0 and len(cluster_tree) > 0:
            eom_clusters = [c for c in is_cluster if is_cluster[c]]
            selected_clusters = []
            # first check if eom_clusters only has root node, which skips epsilon check.
            if (len(eom_clusters) == 1 and eom_clusters[0] == cluster_tree['parent'].min()):
                if allow_single_cluster:
                    selected_clusters = eom_clusters
            else:
                selected_clusters = epsilon_search(set(eom_clusters), cluster_tree, cluster_selection_epsilon, allow_single_cluster)
            for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False

    elif cluster_selection_method == 'leaf':
        leaves = set(get_cluster_tree_leaves(cluster_tree))

        if cluster_selection_epsilon != 0.0:
            selected_clusters = epsilon_search(leaves, cluster_tree, cluster_selection_epsilon, allow_single_cluster)
        else:
            selected_clusters = leaves
        
        # Allow single leaf
        if len(selected_clusters) == 0 and allow_single_cluster:
            for c in is_cluster:
                is_cluster[c] = False
            is_cluster[tree['parent'].min()] = True
        else:
            for c in is_cluster:
                if c in selected_clusters:
                    is_cluster[c] = True
                else:
                    is_cluster[c] = False
    else:
        raise ValueError('Invalid Cluster Selection Method: %s\n'
                         'Should be one of: "eom", "leaf"\n')

    clusters = set([c for c in is_cluster if is_cluster[c]])
    cluster_map = {c: n for n, c in enumerate(sorted(list(clusters)))}
    reverse_cluster_map = {n: c for c, n in cluster_map.items()}

    labels = do_labelling(tree, clusters, cluster_map,
                          allow_single_cluster, cluster_selection_epsilon)
    probs = get_probabilities(tree, reverse_cluster_map, labels)
    stabilities = get_stability_scores(labels, clusters, stability, max_lambda)

    return (labels, probs, stabilities)