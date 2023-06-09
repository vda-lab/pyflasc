"""Predictions functions for FLASC. 

See also :py:mod:`hdbscan.prediction`.
"""
# Original Author: Leland McInnes
# Adapted for FLASC by Jelmer Bot
# License: BSD 3 clause
import numpy as np
from warnings import warn

from hdbscan._hdbscan_tree import recurse_leaf_dfs
from hdbscan.prediction import (
    _find_neighbor_and_lambda,
    get_tree_row_with_child,
)
from ._hdbscan_dist_metrics import DistanceMetric
from ._flasc_depths import _compute_cluster_centralities


def approximate_predict(clusterer, points_to_predict):
    """Predict the cluster label of new points. The returned labels
    will be those of the original clustering found by ``clusterer``,
    and therefore are not (necessarily) the cluster labels that would
    be found by clustering the original data combined with
    ``points_to_predict``, hence the 'approximate' label.

    If you simply wish to assign new points to an existing clustering
    in the 'best' way possible, this is the function to use. If you
    want to predict how ``points_to_predict`` would cluster with
    the original data under FLASC the most efficient existing approach
    is to simply recluster with the new point(s) added to the original dataset.

    Parameters
    ----------
    clusterer : :class:`~flasc.FLASC`
        A clustering object that has been fit to vector inpt data.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    Returns
    -------
    labels : array (n_samples,)
        The predicted labels of the ``points_to_predict``

    probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``

    cluster_labels : array (n_samples,)
        The predicted cluster labels of the ``points_to_predict``

    cluster_probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``

    branch_labels : array (n_samples,)
        The predicted cluster labels of the ``points_to_predict``

    branch_probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``
    """
    if clusterer.hdbscan_._prediction_data is None:
        clusterer.hdbscan_.generate_prediction_data()

    raw_data = clusterer._raw_data
    points_to_predict = np.asarray(points_to_predict)
    if points_to_predict.shape[1] != raw_data.shape[1]:
        raise ValueError("New points dimension does not match fit data!")

    if clusterer.hdbscan_.prediction_data_.cluster_tree.shape[0] == 0:
        warn(
            "Clusterer does not have any defined clusters, new data "
            "will be automatically predicted as noise."
        )
        labels = -1 * np.ones(points_to_predict.shape[0], dtype=np.intp)
        probabilities = np.zeros(points_to_predict.shape[0], dtype=np.double)
        cluster_labels = -1 * np.ones(points_to_predict.shape[0], dtype=np.intp)
        cluster_probabilities = np.zeros(points_to_predict.shape[0], dtype=np.double)
        branch_labels = np.zeros(points_to_predict.shape[0], dtype=np.intp)
        branch_probabilities = np.zeros(points_to_predict.shape[0], dtype=np.double)
        return (
            labels,
            probabilities,
            cluster_labels,
            cluster_probabilities,
            branch_labels,
            branch_probabilities,
        )

    num_predict = points_to_predict.shape[0]
    labels = np.empty(num_predict, dtype=np.intp)
    probabilities = np.zeros(num_predict, dtype=np.double)
    cluster_labels = np.empty(num_predict, dtype=np.intp)
    cluster_probabilities = np.empty(num_predict, dtype=np.double)
    branch_labels = np.zeros(num_predict, dtype=np.intp)
    branch_probabilities = np.ones(num_predict, dtype=np.double)

    min_samples = clusterer.min_samples or clusterer.min_cluster_size
    (
        neighbor_distances,
        neighbor_indices,
    ) = clusterer.hdbscan_.prediction_data_.tree.query(
        points_to_predict, k=2 * min_samples
    )

    for i in range(points_to_predict.shape[0]):
        cluster_label, cluster_prob, nn = _find_cluster_and_probability(
            clusterer.condensed_tree_,
            clusterer.hdbscan_.prediction_data_.cluster_tree,
            neighbor_indices[i],
            neighbor_distances[i],
            clusterer.hdbscan_.prediction_data_.core_distances,
            clusterer.hdbscan_.prediction_data_.cluster_map,
            clusterer.hdbscan_.prediction_data_.max_lambdas,
            min_samples,
        )
        cluster_labels[i] = cluster_label
        cluster_probabilities[i] = cluster_prob

        min_num_branches = 2 if not clusterer.label_sides_as_branches else 1
        if (
            cluster_label >= 0
            and len(clusterer.branch_persistences_[cluster_label]) > min_num_branches
        ):

            labels[i] = clusterer.labels_[nn]
            branch_labels[i] = clusterer.branch_labels_[nn]
            branch_probabilities[i] = clusterer.branch_probabilities_[nn]
            probabilities[i] = (cluster_probabilities[i] + branch_probabilities[i]) / 2
        else:
            labels[i] = cluster_label
            probabilities[i] = cluster_prob

    return (
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        branch_labels,
        branch_probabilities,
    )


def branch_centrality_vectors(clusterer):
    """Predict soft branch-membership vectors for all points in the clusters
    with more than two detected branches.

    Parameters
    ----------
    clusterer : :class:`~flasc.FLASC`
        A clusterer object that has been fit to vector input data.

    Returns
    -------
    centrality_vectors : list[array (n_samples, n_branches)]
        The centrality value of point ``i`` in cluster ``c`` of the original
        dataset from the root of branch ``j`` is in
        ``membership_vectors[c][i, j]``.

    See Also
    --------
    :py:func:`hdbscan.prediction.all_points_membership_vectors`
    """
    if clusterer._raw_data is None:
        raise ValueError("Clusterer object has not been fitted with vector input data!")

    if clusterer.override_cluster_labels is None:
        num_clusters = len(clusterer.cluster_persistence_)
    else:
        num_clusters = len(np.unique(clusterer.cluster_labels_)) - int(
            -1 in clusterer.cluster_labels_
        )
    centrality_vectors = [None for _ in range(num_clusters)]
    for c in range(num_clusters):
        num_branches = len(clusterer.branch_persistences_[c])
        if num_branches <= (1 if clusterer.label_sides_as_branches else 2):
            continue
        points = clusterer.cluster_points_[c]
        graph = clusterer._cluster_approximation_graphs[c].copy()
        branch_points = [
            points[np.where(clusterer.branch_labels_[points] == b)[0]]
            for b in range(len(clusterer.branch_persistences_[c]))
        ]
        branch_centroid = [
            np.average(
                clusterer._raw_data[pts],
                weights=clusterer.branch_probabilities_[pts],
                axis=0,
            )
            for pts in branch_points
        ]
        metric_kwargs = {**clusterer._kwargs}
        if clusterer.metric == "minkowski":
            metric_kwargs.update(p=clusterer.p)
        metric_fun = DistanceMetric.get_metric(clusterer.metric, **metric_kwargs)
        roots = [
            pts[
                int(
                    np.argmin(
                        metric_fun.pairwise(root[None], clusterer._raw_data[pts]),
                        axis=1,
                    )[0]
                )
            ]
            for pts, root in zip(branch_points, branch_centroid)
        ]
        cluster_ids = np.full(clusterer._raw_data.shape[0], -1, dtype=np.double)
        cluster_ids[points] = np.arange(points.shape[0], dtype=np.double)
        graph[:, 0] = cluster_ids[graph[:, 0].astype(np.intp)]
        graph[:, 1] = cluster_ids[graph[:, 1].astype(np.intp)]
        centralities = np.vstack(
            [
                _compute_cluster_centralities(
                    graph, np.intp(cluster_ids[cluster_root]), len(points)
                )
                for cluster_root in roots
            ]
        ).T
        centrality_vectors[c] = centralities
    return centrality_vectors


def update_labels_with_branch_centrality(clusterer, branch_centrality_vectors):
    """Updates the clusterer's labels and branch labels by assigning
    the branch value which has the highest given centrality.

    This can change the label of data points in the center of a cluster.
    These data points are classified as noise by HDBSCAN* in the branch
    detection step, and given the 0-branch-label by default.

    Parameters
    ----------
    clusterer : :class:`~flasc.FLASC`
        A clustering object that has been fit to data.

    branch_centrality_vectors : list[array (n_samples, n_branches)]
        The centrality values from the centroids of a cluster's branches.
        None if the cluster has two or fewer branches.

    Returns
    -------
     labels : np.ndarray, shape (n_samples, )
        Updated cluster+branch labels for each point. Noisy samples are
        given the label -1.

    cluster_labels : np.ndarray, shape (n_samples, )
        Updated luster labels for each point. Noisy samples are given
        the label -1.
    """
    if clusterer.labels_ is None:
        raise ValueError("Clusterer has not been fitted yet!")
    labels = clusterer.labels_.copy()
    branch_labels = clusterer.branch_labels_.copy()
    for points, membership in zip(clusterer.cluster_points_, branch_centrality_vectors):
        if membership is None:
            continue
        label_values = np.unique(clusterer.labels_[points])
        branch_labels[points] = np.argmax(membership, axis=1)
        labels[points] = label_values[branch_labels[points]]
    return labels, branch_labels


def branch_membership_from_centrality(branch_centrality_vectors):
    """Scales the branch centrality vectors to act as probability using softmax.

    .. math::
        \\mathbf{m} = \\frac{
            e^{\\mathbf{c}}
        }{
            \\sum_{i}{e^c_i}
        }

    where :math:`\\mathbf{m}` is the scaled membership vector and
    :math:`\\mathbf{c}` is the branch centrality vector.

    Parameters
    ----------
     branch_centrality_vectors : list[array (n_samples, n_branches)]
        The centrality values from the centroids of a cluster's branches.
        None if the cluster has two or fewer branches.

    Returns
    -------
    scaled_branch_memberships : list[array (n_samples, n_branches)]
        The probabilities of a point belonging to the cluster's branches.
        None if the cluster has two or fewer branches.
    """
    scaled_membership = []
    for branch_centrality in branch_centrality_vectors:
        if branch_centrality is None:
            scaled_membership.append(None)
            continue
        y = np.exp(branch_centrality)
        scaled_membership.append(y / np.sum(y, axis=1)[None].T)
    return scaled_membership


def _find_cluster_and_probability(
    tree,
    cluster_tree,
    neighbor_indices,
    neighbor_distances,
    core_distances,
    cluster_map,
    max_lambdas,
    min_samples,
):
    """
    Return the cluster label (of the original clustering) and membership
    probability of a new data point.

    Parameters
    ----------
    tree : :class:`~hdbscan.plots.CondensedTree`
        The condensed tree associated with the clustering.

    cluster_tree : structured_array
        The raw form of the condensed tree with only cluster information (no
        data on individual points). This is significantly more compact.

    neighbor_indices : array (2 * min_samples, )
        An array of raw distance based nearest neighbor indices.

    neighbor_distances : array (2 * min_samples, )
        An array of raw distances to the nearest neighbors.

    core_distances : array (n_samples, )
        An array of core distances for all points

    cluster_map : dict
        A dictionary mapping cluster numbers in the condensed tree to labels
        in the final selected clustering.

    max_lambdas : dict
        A dictionary mapping cluster numbers in the condensed tree to the
        maximum lambda value seen in that cluster.

    min_samples : int
        The min_samples value used to generate core distances.
    """
    raw_tree = tree._raw_tree
    tree_root = cluster_tree["parent"].min()

    nearest_neighbor, lambda_ = _find_neighbor_and_lambda(
        neighbor_indices, neighbor_distances, core_distances, min_samples
    )

    neighbor_tree_row = get_tree_row_with_child(raw_tree, nearest_neighbor)
    potential_cluster = neighbor_tree_row["parent"]

    if neighbor_tree_row["lambda_val"] > lambda_:
        # Find appropriate cluster based on lambda of new point
        while (
            potential_cluster > tree_root
            and cluster_tree["lambda_val"][cluster_tree["child"] == potential_cluster]
            >= lambda_
        ):
            potential_cluster = cluster_tree["parent"][
                cluster_tree["child"] == potential_cluster
            ][0]

    if potential_cluster in cluster_map:
        cluster_label = cluster_map[potential_cluster]
    else:
        cluster_label = -1

    if cluster_label >= 0:
        max_lambda = max_lambdas[potential_cluster]

        if max_lambda > 0.0:
            lambda_ = min(max_lambda, lambda_)
            prob = lambda_ / max_lambda
        else:
            prob = 1.0
    else:
        prob = 0.0

    return cluster_label, prob, nearest_neighbor


def _find_branch_exemplars(clusterer):
    """Computes branch exemplar points.

    Parameters
    ----------
    clusterer : :class:`~flasc.FLASC`
        A fitted FLASC clusterer object.

    Returns
    -------
    branch_exemplars : list (n_clusters)
        A list with exemplar points for the branches in the clusters. A cluster's
        item is empty if it does not have selected branches. For clusters with
        selected branches, a list with a numpy array of exemplar points for each
        selected branch is given.
    """
    num_clusters = len(clusterer.cluster_condensed_trees_)
    branch_cluster_trees = [
        branch_tree._raw_tree[branch_tree._raw_tree["child_size"] > 1]
        for branch_tree in clusterer.cluster_condensed_trees_
    ]
    selected_branch_ids = [
        sorted(branch_tree._select_clusters())
        for branch_tree in clusterer.cluster_condensed_trees_
    ]
    branch_exemplars = [None] * num_clusters

    for i, points in enumerate(clusterer.cluster_points_):
        selected_branches = selected_branch_ids[i]
        if len(selected_branches) <= (1 if clusterer.label_sides_as_branches else 2):
            continue

        branch_exemplars[i] = []
        raw_condensed_tree = clusterer._cluster_condensed_trees[i]

        for branch in selected_branches:
            _branch_exemplars = np.array([], dtype=np.intp)
            for leaf in recurse_leaf_dfs(branch_cluster_trees[i], branch):
                leaf_max_lambda = raw_condensed_tree["lambda_val"][
                    raw_condensed_tree["parent"] == leaf
                ].max()
                candidates = raw_condensed_tree["child"][
                    (raw_condensed_tree["parent"] == leaf)
                    & (raw_condensed_tree["lambda_val"] == leaf_max_lambda)
                ]
                _branch_exemplars = np.hstack([_branch_exemplars, candidates])
            ids = points[_branch_exemplars]
            branch_exemplars[i].append(clusterer._raw_data[ids, :])

    return branch_exemplars
