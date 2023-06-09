# sklearn FLASC clusterer
# Author: Jelmer Bot
# Closely based on HDBSCAN clusterer
# License: BSD 3 clause
import numpy as np
from joblib import Memory
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, ClusterMixin
from hdbscan import HDBSCAN
from hdbscan.hdbscan_ import (
    is_finite,
    get_finite_row_indices,
    check_precomputed_distance_matrix,
    remap_condensed_tree,
    remap_single_linkage_tree,
)
from hdbscan.plots import CondensedTree, SingleLinkageTree, MinimumSpanningTree

from ._flasc import flasc
from .plots import ApproximationGraph
from .prediction import _find_branch_exemplars


class FLASC(BaseEstimator, ClusterMixin):
    """Performs hdbscan clustering with flare detection post-processing step.

    FLASC - Flare-Sensitive Clustering.
    Performs :py:mod:`hdbscan` clustering [1]_ with a post-processing step to 
    detect branches within individual clusters. For each cluster, a graph is
    constructed connecting the data points based on their mutual reachability
    distances. Each edge is given a centrality value based on how many edges
    need to be traversed to reach the cluster's root point from the edge. Then,
    the edges are clustered as if that centrality was a density, progressively
    removing the 'centre' of each cluster and seeing how many branches remain.

    Parameters
    ----------
    X : array of shape (n_samples, n_features), or \
        array of shape (n_samples, n_samples)
        A feature array, or array of distances between samples if
        ``metric='precomputed'``.

    min_cluster_size : int, optional (default=5)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_branch_size : int, optional (default=None)
        The minimum number of samples in a group for that group to be
        considered a branch; groupings smaller than this size will seen as
        points falling out of a branch. Defaults to the min_cluster_size.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        Defaults to the min_cluster_size.

    metric : str or callable, optional (default='minkowski')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

    p : int, optional (default=2)
        p value to use if using the minkowski metric.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.
        See [2]_ for more information.

    algorithm : str, optional (default='best')
        Exactly which algorithm to use; hdbscan has variants specialised
        for different characteristics of the data. By default this is set
        to ``best`` which chooses the "best" algorithm given the nature of
        the data. You can force other options if you believe you know
        better. Options are:
          * ``best``
          * ``generic``
          * ``prims_kdtree``
          * ``prims_balltree``
          * ``boruvka_kdtree``
          * ``boruvka_balltree``

    leaf_size : int, optional (default=40)
        Leaf size for trees responsible for fast nearest
        neighbour queries.

    approx_min_span_tree : bool, optional (default=True)
        Whether to accept an only approximate minimum spanning tree.
        For some algorithms this can provide a significant speedup, but
        the resulting clustering may be of marginally lower quality.
        If you are willing to sacrifice speed for correctness you may want
        to explore this; in general this should be left at the default True.

    cluster_selection_method : str, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for FLASC is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:
          * ``eom``
          * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to t=True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
        (default False)

    cluster_selection_epsilon: float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        See [3]_ for more information. Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using :func:`~flasc.predictions.approximate_predict`), as the 
        that function is not aware of this argument.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future 
        (e.g. using :func:`~flasc.predictions.approximate_predict`), as
        the approximate_predict function is not aware of this argument.

    allow_single_branch : bool, optional (default=False)
        Analogous to ``allow_single_cluster``. Note that depending on
        ``label_sides_as_branches`` FLASC* requires at least 3 branches to
        exist in a cluster before they are incorporated in the final labelling.

    branch_detection_method : str, optional (default=``full``)
        Deteremines which graph is conctructed to detect branches with. Valid
        values are, ordered by increasing computation cost and decreasing
        sensitivity to noise:
         - ``core``: Contains the edges that connect each point to all other 
          points within a mutual reachability distance lower than or equal to
          the point's core distance. This is the cluster's subgraph of the 
          k-NN graph over the entire data set (with k = ``min_samples``).
        - ``full``: Contains all edges between points in each cluster with a 
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.
        
    branch_selection_persistence: float, optional (default=0.0)
        A centrality persistence threshold. Branches with a persistence below 
        this value will be merged. See [3]_ for more information. Note that this
        should not be used if we want to predict the cluster labels for new 
        points in future (e.g. using approximate_predict), as the 
        :func:`~flasc.prediction.approximate_predict` function
        is not aware of this argument.
        
    branch_selection_method : str, optional (default='eom')
        The method used to select branches from the cluster's condensed tree.
        The standard approach for FLASC* is to use the ``eom`` approach.
        Options are:
          * ``eom``
          * ``leaf``

    max_branch_size : int, optional (default=0)
        A limit to the size of clusters returned by the ``eom`` algorithm.
        Has no effect when using ``leaf`` clustering (where clusters are
        usually small regardless). Note that this should not be used if we
        want to predict the cluster labels for new points in future (e.g. using
        :func:`~flasc.predictions.approximate_predict`), as that function is
        not aware of this argument.

    label_sides_as_branches : bool, optional (default=False),
        When this flag is False, branches are only labelled for clusters with at
        least three branches (i.e., at least y-shapes). Clusters with only two 
        branches represent l-shapes. The two branches describe the cluster's
        outsides growing towards each other. Enableing this flag separates these
        branches from each other in the produced labelling.

    override_cluster_labels : np.ndarray, optional (default=None)
        Override the HDBSCAN* clustering to specify your own grouping with a
        numpy array containing a cluster label for each data point. Negative
        values will be interpreted as noise points. When the parameter is not set
        to None, core distances are computed over all data points,
        minimum spanning trees and the branches are computed per cluster. 
        Consequently, the manually specified clusters do not have to form 
        neatly separable connected components in the minimum spanning tree 
        over all the data points. 
        
        Because the clustering step is skipped, some of the attributes
        and the :func:`~flasc.prediction.approximate_predict` function will
        be unavailable:
        - cluster_persistence_
        - condensed_tree_
        - single_linkage_tree_
        - min_spanning_tree_
        - cluster_exemplars_
    
    override_cluster_probabilities : np.ndarray, optional (default=None)
        Specifying a not None value for this parameter is only valid when
        ``override_cluster_labels`` is used. In that case, this parameter 
        controls the data point cluster membership probabilities. When this 
        parameter is None, a default 1.0 probability is used for all points.
        
    memory : instance of joblib.Memory or str, optional
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    num_jobs : int, optional (default=None)
        Number of parallel jobs to run in core distance computations and branch 
        detection step. For ``num_jobs`` below -1, (n_cpus + 1 + num_jobs) are 
        used. By default, the algorithm tries to estimate whether the given input
        is large enough for multi-processing to have a benefit. If so, 4 processes
        are started, otherwise only the main process is used. When a ``num_jobs`` 
        value is given, that number of jobs is used regardless of the input size.

    **kwargs : optional
        Additional arguments passed to :func:`~hdbscan.hdbscan` or the 
        distance metric.

    Attributes
    ----------
    labels_ : np.ndarray, shape (n_samples, )
        Cluster+branch labels for each point. Noisy samples are given the
        label -1.

    probabilities_ : np.ndarray, shape (n_samples, )
        Cluster+branch membership strengths for each point. Noisy samples are
        assigned 0.

    cluster_labels_ : np.ndarray, shape (n_samples, )
        Cluster labels for each point. Noisy samples are given the label -1.

    cluster_probabilities_ : np.ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are
        assigned 0.
    
    cluster_persistences_ : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores gauge the relative coherence of the clusters output by the 
        algorithm. Not available when ``override_cluster_labels`` is used.

    branch_labels_ : np.ndarray, shape (n_samples, )
        Branch labels for each point. Noisy samples are given the label -1.

    branch_probabilities_ : np.ndarray, shape (n_samples, )
        Branch membership strengths for each point. Noisy samples are
        assigned 0.

    branch_persistences_ : tuple (n_clusters)
        A branch persistence for each cluster produced during the branch
        detection step.

    condensed_tree_ : :class:`~hdbscan.plots.CondensedTree`
        The condensed tree hierarchy used to generate clusters.
        Not available when ``override_cluster_labels`` is used.

    single_linkage_tree_ : :class:`~hdbscan.plots.SingleLinkageTree`
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Not available when ``override_cluster_labels`` is used.

    min_spanning_tree_ : :class:`~hdbscan.plots.MinimumSpanningTree`
        The minimum spanning as an edgelist. Not available when 
        ``override_cluster_labels`` is used.
    
    cluster_approximation_graph_ : :class:`~flasc.plots.ApproximationGraph`
        The graphs used to detect branches in each cluster as an 
        :class:`~flasc.plots.ApproximationGraph`. Can be converted to
        a networkx graph, pandas data frame, or a list with numpy array-edgelists.
        Points are labelled by their row-index into the input data. The edges 
        contained in the graph depend on the ``branch_detection_method``:
        - ``core``: Contains the edges that connect each point to all other 
          points in a cluster within a mutual reachability distance lower than 
          or equal to the point's core distance. This is an extension of the 
          minimum spanning tree introducing only edges with equal distances. The
          reachability distance introduces ``num_points`` * ``min_samples`` of 
          such edges.
        - ``full``: Contains all edges between points in each cluster with a 
          mutual reachability distance lower than or equal to the distance of
          the most-distance point in each cluster. These graphs represent the
          0-dimensional simplicial complex of each cluster at the first point in
          the filtration where they contain all their points.

    cluster_condensed_trees : list[:class:`~hdbscan.plots.CondensedTree`]
        Condensed hierarchies for each cluster produced during the branch
        detection step. Data points are numbered with in-cluster ids.

    cluster_linkage_trees_ : list[:class:`~hdbscan.plots.SingleLinkageTree`]
        Single linkage trees for each cluster produced during the branch
        detection step, in the scipy hierarchical clustering format.
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Data points are numbered with in-cluster ids.

    cluster_centralities_ : np.ndarray, shape (n_samples, )
        Centrality values for each point in a cluster. Overemphasizes points' 
        eccentricity within the cluster as the values are based on minimum 
        spanning trees that do not contain the equally distanced edges resulting
        from the mutual reachability distance. 
        
    cluster_points_ : list (n_clusters)
        The data point row indices for each detected cluster:
        ``cluster_points_[l] = np.where(cluster_labels == l)[0]``

    cluster_exemplars_ : list
        A list of exemplar points for clusters. Since HDBSCAN supports
        arbitrary shapes for clusters we cannot provide a single cluster
        exemplar per cluster. Instead a list is returned with each element
        of the list being a numpy array of exemplar points for a cluster --
        these points are the "most representative" points of the cluster.
        Not available when ``override_cluster_labels`` is used or a precomputed
        distance matrix is given as input.

    branch_exemplars_ : list (n_clusters)
        A list with exemplar points for the branches in the clusters. A cluster's
        item is empty if it does not have selected branches. For clusters with 
        selected branches, a list with a numpy array of exemplar points for each
        selected branch is given.

    relative_validity_ : float
        HDBSCAN's fast approximation of the Density Based Cluster Validity (
        DBCV) score [4]_ on FLASC's labelling. It may only be used to compare
        results across different choices of hyper-parameters, therefore 
        is only a relative score.

    hdbscan_ : :class:`~hdbscan.HDBSCAN`
        An HDBSCAN clusterer object fitted to the data. Can be used to compute
        outlier scores and cluster exemplars. Not available when 
        ``override_cluster_labels`` is used.

    References
    ----------
    
    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates.
       In Pacific-Asia Conference on Knowledge Discovery and Data Mining
       (pp. 160-172). Springer Berlin Heidelberg.

    .. [2] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    .. [3] Malzer, C., & Baum, M. (2019). A Hybrid Approach To Hierarchical
       Density-based Cluster Selection. arxiv preprint 1911.02282.

    .. [4] Moulavi, D., Jaskowiak, P.A., Campello, R.J., Zimek, A. and
       Sander, J., 2014. Density-Based Clustering Validation. In SDM
       (pp. 839-847).
    """

    def __init__(
        self,
        min_cluster_size=5,
        min_branch_size=None,
        min_samples=None,
        metric="minkowski",
        p=2,
        alpha=1.0,
        algorithm="best",
        leaf_size=40,
        approx_min_span_tree=True,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        cluster_selection_epsilon=0.0,
        max_cluster_size=0,
        allow_single_branch=False,
        branch_detection_method="full",
        branch_selection_method="eom",
        branch_selection_persistence=0.0,
        max_branch_size=0,
        label_sides_as_branches=False,
        override_cluster_labels=None,
        override_cluster_probabilities=None,
        memory=Memory(None, verbose=0),
        num_jobs=None,
        **kwargs
    ):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        # Parameters
        self.min_cluster_size = min_cluster_size
        self.min_branch_size = min_branch_size
        self.min_samples = min_samples
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.approx_min_span_tree = approx_min_span_tree
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.allow_single_branch = allow_single_branch
        self.branch_detection_method = branch_detection_method
        self.branch_selection_method = branch_selection_method
        self.branch_selection_persistence = branch_selection_persistence
        self.max_branch_size = max_branch_size
        self.label_sides_as_branches = label_sides_as_branches
        self.override_cluster_labels = override_cluster_labels
        self.override_cluster_probabilities = override_cluster_probabilities
        self.memory = memory
        self.num_jobs = num_jobs
        self._kwargs = kwargs

        # Outputs
        self.labels_ = None
        self.probabilities_ = None
        self.cluster_labels_ = None
        self.cluster_probabilities_ = None
        self.cluster_persistences_ = None
        self.branch_labels_ = None
        self.branch_probabilities_ = None
        self.branch_persistences_ = None
        self._condensed_tree = None
        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._cluster_approximation_graphs = None
        self._cluster_condensed_trees = None
        self._cluster_linkage_trees = None
        self.cluster_centralities_ = None
        self.cluster_points_ = None
        self._all_finite = None
        self._raw_data = None

        # Properties
        self._relative_validity = None
        self._branch_exemplars = None
        self._hdbscan = None

    def fit(self, X: np.ndarray, y=None):
        """Performs the branch aware clustering.

        Parameters
        ----------
        X : array of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
          A feature array, or array of distances between samples if
          ``metric='precomputed'``.
        y : Not used

        Returns
        -------
        self : object
          Returns self
        """
        if self.metric != "precomputed":
            X = check_array(X, force_all_finite=False)
            self._raw_data = X.astype(np.double)

            self._all_finite = is_finite(X)
            if not self._all_finite:
                finite_index = get_finite_row_indices(X)
                clean_data = X[finite_index]
                internal_to_raw = {
                    x: y for x, y in zip(range(len(finite_index)), finite_index)
                }
                outliers = list(set(range(X.shape[0])) - set(finite_index))
            else:
                clean_data = X
        elif issparse(X):
            X = check_array(X)
            clean_data = X
        else:
            check_precomputed_distance_matrix(X)
            clean_data = X

        kwargs = self.get_params()
        kwargs.update(self._kwargs)
        if self.metric != "precomputed" and not self._all_finite:
            if self.override_cluster_labels is not None:
                kwargs.update(
                    override_cluster_labels=self.override_cluster_labels[finite_index]
                )
            if self.override_cluster_probabilities is not None:
                kwargs.update(
                    override_cluster_probabilities=self.override_cluster_probabilities[
                        finite_index
                    ]
                )
        (
            # Combined result
            self.labels_,
            self.probabilities_,
            # Clustering result
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.cluster_persistence_,
            # Branching result
            self.branch_labels_,
            self.branch_probabilities_,
            self.branch_persistences_,
            # Data to clusters
            self._condensed_tree,
            self._single_linkage_tree,
            self._min_spanning_tree,
            # Clusters to branches
            self._cluster_approximation_graphs,
            self._cluster_condensed_trees,
            self._cluster_linkage_trees,
            self.cluster_centralities_,
            self.cluster_points_,
        ) = flasc(clean_data, **kwargs)

        if self.metric != "precomputed" and not self._all_finite:
            if self.override_cluster_labels is None:
                self._condensed_tree = remap_condensed_tree(
                    self._condensed_tree, internal_to_raw, outliers
                )
                self._single_linkage_tree = remap_single_linkage_tree(
                    self._single_linkage_tree, internal_to_raw, outliers
                )

            _remap_point_lists(self.cluster_points_, internal_to_raw)
            _remap_edge_lists(self._cluster_approximation_graphs, internal_to_raw)

            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.labels_
            self.labels_ = new_labels

            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.cluster_labels_
            self.cluster_labels_ = new_labels

            new_labels = np.full(X.shape[0], -1)
            new_labels[finite_index] = self.branch_labels_
            self.branch_labels_ = new_labels

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.probabilities_
            self.probabilities_ = new_probabilities

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.cluster_probabilities_
            self.cluster_probabilities_ = new_probabilities

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.branch_probabilities_
            self.branch_probabilities_ = new_probabilities

            new_probabilities = np.zeros(X.shape[0])
            new_probabilities[finite_index] = self.cluster_centralities_
            self.cluster_centralities_ = new_probabilities

        return self

    def fit_predict(self, X, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        X : array of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
          A feature array, or array of distances between samples if
          ``metric='precomputed'``.
        y : not used
        Returns
        -------
        y : np.ndarray, shape (n_samples, )
            cluster labels.
        """
        self.fit(X, y)
        return self.labels_

    def weighted_centroid(self, label_id, data=None):
        """Provides an approximate representative point for a given branch.
        Note that this technique assumes a euclidean metric for speed of
        computation. For more general metrics use the
        :func:`~flasc._sklearn.FLASC.weighted_medoid`
        method which is slower, but can work with the metric the model trained
        with.

        Parameters
        ----------
        label_id: int
            The id of the cluster to compute a centroid for.

        data : np.ndarray (n_samples, n_features), optional (default=None)
            A dataset to use instead of the raw data that was clustered on.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative centroid for cluster ``label_id``.
        """
        if self.labels_ is None:
            raise AttributeError("Model has not been fit to data")
        if self._raw_data is None and data is None:
            raise AttributeError("Raw data not available")
        if label_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )
        if data is None:
            data = self._raw_data
        mask = self.labels_ == label_id
        cluster_data = data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        return np.average(cluster_data, weights=cluster_membership_strengths, axis=0)

    def weighted_cluster_centroid(self, cluster_id):
        """Wraps :func:`hdbscan.HDBSCAN.weighted_cluster_centroid`.

        Provides an approximate representative point for a given cluster. Note
        that this technique assumes a euclidean metric for speed of computation.
        For more general metrics use the
        :func:`~flasc._sklearn.FLASC.weighted_cluster_medoid`
        method which is slower, but can work with the metric the model trained
        with.

        Parameters
        ----------
        cluster_id: int
            The id of the cluster to compute a centroid for.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative centroid for cluster ``cluster_id``.
        """
        return self.hdbscan_.weighted_cluster_centroid(cluster_id)

    def weighted_medoid(self, label_id, data=None):
        """Provides an approximate representative point for a given branch.

        Note that this technique can be very slow and memory intensive for
        large clusters. For faster results use the
        :func:`~flasc._sklearn.FLASC.weighted_centroid`
        method which is faster, but assumes a euclidean metric.

        Parameters
        ----------
        label_id: int
            The id of the cluster to compute a medoid for.

        data : np.ndarray (n_samples, n_features), optional (default=None)
            A dataset to use instead of the raw data that was clustered on.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative medoid for cluster ``label_id``.
        """
        if self.labels_ is None:
            raise AttributeError("Model has not been fit to data")
        if self._raw_data is None and data is None:
            raise AttributeError("Raw data not available")
        if label_id == -1:
            raise ValueError(
                "Cannot calculate weighted centroid for -1 cluster "
                "since it is a noise cluster"
            )
        if data is None:
            data = self._raw_data
        mask = self.labels_ == label_id
        cluster_data = data[mask]
        cluster_membership_strengths = self.probabilities_[mask]

        dist_mat = pairwise_distances(cluster_data, metric=self.metric, **self._kwargs)

        dist_mat = dist_mat * cluster_membership_strengths
        medoid_index = np.argmin(dist_mat.sum(axis=1))
        return cluster_data[medoid_index]

    def weighted_cluster_medoid(self, cluster_id):
        """Wraps :func:`hdbscan.HDBSCAN.weighted_cluster_mentroid`.

        Provides an approximate representative point for a given cluster.
        Note that this technique can be very slow and memory intensive for
        large clusters. For faster results use the
        :func:`~flasc._sklearn.FLASC.weighted_cluster_centroid`
        method which is faster, but assumes a euclidean metric.

        Parameters
        ----------
        cluster_id: int
            The id of the cluster to compute a medoid for.

        Returns
        -------
        centroid: array of shape (n_features,)
            A representative medoid for cluster ``cluster_id``.
        """
        return self.hdbscan_.weighted_cluster_medoid(cluster_id)

    @property
    def condensed_tree_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self.override_cluster_labels is not None:
            raise AttributeError(
                "HDBSCAN object not available with overridden clusters."
            )
        if self._condensed_tree is None:
            raise AttributeError(
                "No condensed tree was generated; try running fit first."
            )

        return CondensedTree(
            self._condensed_tree,
            self.cluster_selection_method,
            self.allow_single_cluster,
        )

    @property
    def single_linkage_tree_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self.override_cluster_labels is not None:
            raise AttributeError(
                "Single linkage tree not available with overridden clusters."
            )
        if self._single_linkage_tree is None:
            raise AttributeError(
                "No single linkage tree was generated; try running fit first."
            )
        return SingleLinkageTree(self._single_linkage_tree)

    @property
    def minimum_spanning_tree_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self.override_cluster_labels is not None:
            raise AttributeError(
                "Minimum spanning tree not available with overridden clusters."
            )
        if self._raw_data is None:
            raise AttributeError(
                "Minimum spanning tree not available with precomputed " "distances."
            )
        if self._min_spanning_tree is None:
            raise AttributeError(
                "No  minimum spanning tree was generated; try running fit first."
            )
        return MinimumSpanningTree(self._min_spanning_tree, self._raw_data)

    @property
    def cluster_approximation_graph_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._cluster_approximation_graphs is None:
            raise AttributeError(
                "No cluster approximation graph was generated; try running fit first."
            )
        return ApproximationGraph(
            self._cluster_approximation_graphs,
            self.labels_,
            self.probabilities_,
            self.cluster_labels_,
            self.cluster_probabilities_,
            self.cluster_centralities_,
            self.branch_labels_,
            self.branch_probabilities_,
            self._raw_data,
        )

    @property
    def cluster_condensed_trees_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._cluster_condensed_trees is None:
            raise AttributeError(
                "No cluster condensed trees were generated; try running fit first."
            )
        return [
            CondensedTree(tree, self.branch_selection_method, self.allow_single_branch)
            for tree in self._cluster_condensed_trees
        ]

    @property
    def cluster_linkage_trees_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._cluster_linkage_trees is None:
            raise AttributeError(
                "No cluster linkage trees were generated; try running fit first."
            )
        return [
            SingleLinkageTree(tree)
            for tree in self._cluster_linkage_trees
        ]

    @property
    def branch_exemplars_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._branch_exemplars is not None:
            return self._branch_exemplars
        if self._raw_data is None:
            raise AttributeError(
                "Branch exemplars not available with precomputed " "distances."
            )
        if self._cluster_condensed_trees is None:
            raise AttributeError("No branches detected; try running fit first.")
        self._branch_exemplars = _find_branch_exemplars(self)
        return self._branch_exemplars

    @property
    def cluster_exemplars_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._raw_data is None:
            raise AttributeError(
                "Cluster exemplars not available with precomputed " "distances."
            )
        if self.override_cluster_labels is not None:
            raise AttributeError(
                "Cluster exemplars not available with overridden clusters."
            )
        if self._condensed_tree is None:
            raise AttributeError("Cluster not detected yet; try running fit first.")
        if self.hdbscan_._prediction_data is None:
            self.hdbscan_.generate_prediction_data()
        return self.hdbscan_.prediction_data_.exemplars

    @property
    def relative_validity_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._relative_validity is not None:
            return self._relative_validity

        if self._raw_data is None:
            raise AttributeError(
                "Relative validity is not available with precomputed " "distances."
            )
        if self.override_cluster_labels is not None:
            raise AttributeError(
                "Relative validity isnot available with overridden clusters."
            )
        if self.labels_ is None:
            raise AttributeError("Cluster not detected yet; try running fit first.")

        labels = self.labels_
        sizes = np.bincount(labels + 1)
        noise_size = sizes[0]
        cluster_size = sizes[1:]
        total = noise_size + np.sum(cluster_size)
        num_clusters = len(cluster_size)
        DSC = np.zeros(num_clusters)
        min_outlier_sep = np.inf  # only required if num_clusters = 1
        correction_const = 2  # only required if num_clusters = 1

        # Unltimately, for each Ci, we only require the
        # minimum of DSPC(Ci, Cj) over all Cj != Ci.
        # So let's call this value DSPC_wrt(Ci), i.e.
        # density separation 'with respect to' Ci.
        DSPC_wrt = np.ones(num_clusters) * np.inf
        max_distance = 0

        mst_df = self.minimum_spanning_tree_.to_pandas()

        for edge in mst_df.iterrows():
            label1 = labels[int(edge[1]["from"])]
            label2 = labels[int(edge[1]["to"])]
            length = edge[1]["distance"]

            max_distance = max(max_distance, length)

            if label1 == -1 and label2 == -1:
                continue
            elif label1 == -1 or label2 == -1:
                # If exactly one of the points is noise
                min_outlier_sep = min(min_outlier_sep, length)
                continue

            if label1 == label2:
                # Set the density sparseness of the cluster
                # to the sparsest value seen so far.
                DSC[label1] = max(length, DSC[label1])
            else:
                # Check whether density separations with
                # respect to each of these clusters can
                # be reduced.
                DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
                DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

        # In case min_outlier_sep is still np.inf, we assign a new value to it.
        # This only makes sense if num_clusters = 1 since it has turned out
        # that the MR-MST has no edges between a noise point and a core point.
        min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

        # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
        # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
        # MR-MST might contain an edge with one point in Cj and ther other one
        # in Ck. Here, we replace the infinite density separation of Ci by
        # another large enough value.
        #
        # TODO: Think of a better yet efficient way to handle this.
        correction = correction_const * (
            max_distance if num_clusters > 1 else min_outlier_sep
        )
        DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

        V_index = [
            (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i])
            for i in range(num_clusters)
        ]
        score = np.sum(
            [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
        )
        self._relative_validity = score
        return self._relative_validity

    @property
    def hdbscan_(self):
        """See :class:`~flasc._sklearn.FLASC` for documentation."""
        if self._hdbscan is not None:
            return self._hdbscan

        if self.override_cluster_labels is not None:
            raise AttributeError(
                "HDBSCAN object not available with overridden clusters."
            )

        if self.labels_ is None:
            raise AttributeError("Clusters not detected yet; try running fit first!")

        self._hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            p=self.p,
            alpha=self.alpha,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            approx_min_span_tree=self.approx_min_span_tree,
            cluster_selection_method=self.cluster_selection_method,
            allow_single_cluster=self.allow_single_cluster,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            max_cluster_size=self.max_cluster_size,
            memory=self.memory,
            core_dist_n_jobs=4 if self.num_jobs is None else self.num_jobs,
            gen_min_span_tree=True,
            match_reference_implementation=False,
            **self._kwargs
        )
        self._hdbscan.labels_ = self.cluster_labels_
        self._hdbscan.probabilities_ = self.cluster_probabilities_
        self._hdbscan.cluster_persistence_ = self.cluster_persistence_
        self._hdbscan._condensed_tree = self._condensed_tree
        self._hdbscan._single_linkage_tree = self._single_linkage_tree
        self._hdbscan._min_spanning_tree = self._min_spanning_tree
        self._hdbscan._all_finite = self._all_finite
        self._hdbscan._raw_data = self._raw_data
        return self._hdbscan


def _remap_edge_lists(edge_lists, internal_to_raw):
    """
    Takes a list of edge lists and replaces the internal indices to raw indices.

    Parameters
    ----------
    edge_lists : list[np.ndarray]
        A list of numpy edgelists with the first two columns indicating
        datapoints.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index.
    """
    for graph in edge_lists:
        for edge in graph:
            edge[0] = internal_to_raw[edge[0]]
            edge[1] = internal_to_raw[edge[1]]


def _remap_point_lists(point_lists, internal_to_raw):
    """
    Takes a list of points lists and replaces the internal indices to raw indices.

    Parameters
    ----------
    point_lists : list[np.ndarray]
        A list of numpy arrays with point indices.
    internal_to_raw: dict
        A mapping from internal integer index to the raw integer index.
    """
    for points in point_lists:
        for idx in range(len(points)):
            points[idx] = internal_to_raw[points[idx]]
