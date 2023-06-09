# Main flasc function
# Author: Jelmer Bot
# Closely based on the hdbscan function.
# License: BSD 3 clause
import numpy as np
from joblib import Memory
from joblib.parallel import Parallel, cpu_count
from scipy.sparse import issparse
from sklearn.utils import check_array
from sklearn.neighbors import KDTree, BallTree
from hdbscan.hdbscan_ import check_precomputed_distance_matrix

from ._hdbscan import (
    _hdbscan_generic_reachability,
    _hdbscan_generic_linkage,
    _hdbscan_space_tree,
    _hdbscan_space_tree_linkage,
    _hdbscan_space_tree_core_dists_prims,
    _hdbscan_extract_clusters,
)
from ._flasc_branches import (
    _extract_cluster_points,
    _extract_cluster_msts,
    _update_labelling,
)
from ._flasc_threaded import (
    _compute_msts_in_cluster_generic,
    _compute_msts_in_cluster_space_tree,
    _compute_branch_linkage,
    _compute_branch_segmentation,
)

FAST_METRICS = (
    KDTree.valid_metrics + BallTree.valid_metrics + ["cosine", "arccos"]
)


def flasc(
    X,
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
    **kwargs,
):
    """Performs FLASC clustering with flare detection post-processing step.

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
        By default FLASC will not produce a single cluster, setting this
        to t=True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.
        (default False)

    cluster_selection_epsilon: float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        See [3]_ for more information. Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using approximate_predict), as the approximate_predict function
        is not aware of this argument.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.

    allow_single_branch : bool, optional (default=False)
        Analogous to ``allow_single_cluster``. Note that depending on
        ``label_sides_as_branches`` FFLASC requires at least 3 branches to
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
        
    branch_selection_method : str, optional (default='eom')
        The method used to select branches from the cluster's condensed tree.
        The standard approach for FFLASC is to use the ``eom`` approach.
        Options are:
          * ``eom``
          * ``leaf``

    branch_selection_persistence: float, optional (default=0.0)
        A centrality persistence threshold. Branches with a persistence below 
        this value will be merged. See [3]_ for more information. Note that this
        should not be used if we want to predict the cluster labels for new 
        points in future (e.g. using approximate_predict), as the 
        :func:`~flasc.prediction.approximate_predict` function
        is not aware of this argument.

    max_branch_size : int, optional (default=0)
        A limit to the size of clusters returned by the ``eom`` algorithm.
        Has no effect when using ``leaf`` clustering (where clusters are
        usually small regardless). Note that this should not be used if we
        want to predict the cluster labels for new points in future (e.g. using
        :func:`~flasc.prediction.approximate_predict`), as that function is
        not aware of this argument.

    label_sides_as_branches : bool, optional (default=False),
        When this flag is False, branches are only labelled for clusters with at
        least three branches (i.e., at least y-shapes). Clusters with only two 
        branches represent l-shapes. The two branches describe the cluster's
        outsides growing towards each other. Enableing this flag separates these
        branches from each other in the produced labelling.

    override_cluster_labels : np.ndarray, optional (default=None)
        Override the FLASC clustering to specify your own grouping with a
        numpy array containing a cluster label for each data point. Negative
        values will be interpreted as noise points. When the parameter is not set
        to None, core distances are computed over all data points,
        minimum spanning trees and the branches are computed per cluster. 
        Consequently, the manually specified clusters do not have to form 
        neatly separable connected components in the minimum spanning tree 
        over all the data points. 
        
        Because the clustering step is skipped, some of the output variables 
        and the :func:`~flasc.prediction.approximate_predict` function will
        be unavailable:
        - cluster_persistence
        - condensed_tree
        - single_linkage_tree
        - min_spanning_tree
    
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

    Returns
    -------
    labels : np.ndarray, shape (n_samples, )
        Cluster+branch labels for each point. Noisy samples are given the
        label -1.

    probabilities : np.ndarray, shape (n_samples, )
        Cluster+branch membership strengths for each point. Noisy samples are
        assigned 0.

    cluster_labels : np.ndarray, shape (n_samples, )
        Cluster labels for each point. Noisy samples are given the label -1.

    cluster_probabilities : np.ndarray, shape (n_samples, )
        Cluster membership strengths for each point. Noisy samples are
        assigned 0.
    
    cluster_persistence : array, shape  (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores gauge the relative coherence of the clusters output by the 
        algorithm. Not available when ``override_cluster_labels`` is used.

    branch_labels : np.ndarray, shape (n_samples, )
        Branch labels for each point. Noisy samples are given the label -1.

    branch_probabilities : np.ndarray, shape (n_samples, )
        Branch membership strengths for each point. Noisy samples are
        assigned 0.

    branch_persistences : tuple (n_clusters)
        A branch persistence for each cluster produced during the branch
        detection step.

    condensed_tree : record array
        The condensed cluster hierarchy used to generate clusters.
        Not available when ``override_cluster_labels`` is used.

    single_linkage_tree : np.ndarray, shape (n_samples - 1, 4)
        The single linkage tree produced during clustering in scipy
        hierarchical clustering format
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Not available when ``override_cluster_labels`` is used.

    min_spanning_tree : np.ndarray, shape (n_samples - 1, 3)
        The minimum spanning as an edgelist. Not available when 
        ``override_cluster_labels`` is used.

    cluster_approximation_graphs : tuple (n_clusters)
        The graphs used to detect branches in each cluster stored as a numpy 
        array with four columns: source, target, centrality, mutual reachability
        distance. Points are labelled by their row-index into the input data. 
        The edges contained in the graphs depend on the ``branch_detection_method``:
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

    cluster_condensed_trees : tuple (n_clusters)
        A condensed branch hierarchy for each cluster produced during the
        branch detection step. Data points are numbered with in-cluster ids.

    cluster_linkage_trees : tuple (n_clusters)
        A single linkage tree for each cluster produced during the branch
        detection step, in the scipy hierarchical clustering format.
        (see http://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html).
        Data points are numbered with in-cluster ids.

    cluster_centralities : np.ndarray, shape (n_samples, )
        Centrality values for each point in a cluster. Overemphasizes points' 
        eccentricity within the cluster as the values are based on minimum 
        spanning trees that do not contain the equally distanced edges resulting
        from the mutual reachability distance. 

    cluster_points : list (n_clusters)
        The data point row indices for each cluster.
    
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
    """
    # - Validate input values, might change values.

    # -- Check integer / floating values
    def __check_greater_equal_integer(min_value=1, **kwargs):
        for k, v in kwargs.items():
            if not (np.issubdtype(type(v), np.integer) and v >= min_value):
                raise ValueError(
                    f"``{k}`` must be an integer greater or equal "
                    f"to {min_value}, {v} given."
                )

    def __check_greater_equal_float(min_value=0.0, **kwargs):
        for k, v in kwargs.items():
            if not (np.issubdtype(type(v), np.floating) and v >= min_value):
                raise ValueError(
                    f"``{k}`` must be a float greater or equal to "
                    f"{min_value}, {v} given."
                )

    # --- Fill in default values.
    if min_branch_size is None:
        min_branch_size = min_cluster_size
    if min_samples is None:
        min_samples = min_cluster_size

    # --- Set alpha and cluster_selection to float without error.
    alpha = float(alpha)
    cluster_selection_epsilon = float(cluster_selection_epsilon)
    branch_selection_persistence = float(branch_selection_persistence)

    # --- Check that values are positive
    __check_greater_equal_integer(
        min_value=2,
        min_cluster_size=min_cluster_size,
        min_branch_size=min_branch_size,
    )
    __check_greater_equal_integer(
        min_value=1,
        min_samples=min_samples,
        leaf_size=leaf_size,
    )
    __check_greater_equal_float(
        cluster_selection_epsilon=cluster_selection_epsilon,
        branch_selection_persistence=branch_selection_persistence,
        alpha=alpha,
    )

    # -- Metric
    if metric == "minkowski":
        if p < 0:
            raise ValueError(
                "Minkowski metric with negative p value is not defined."
            )
        # Move p to kwargs
        kwargs["p"] = p

    # -- Algorithm
    if algorithm not in [
        "best",
        "generic",
        "prims_kdtree",
        "prims_balltree",
        "boruvka_kdtree",
        "boruvka_balltree",
    ]:
        raise TypeError(f"Unknown algorithm specified, {algorithm} given.")
    if algorithm.endswith("kdtree") and metric not in KDTree.valid_metrics:
        raise ValueError("Cannot use a KDTree for this metric.")
    if algorithm.endswith("balltree") and metric not in KDTree.valid_metrics:
        raise ValueError("Cannot use a BallTree for this metric.")

    # -- Match reference implementation
    if "match_reference_implementation" in kwargs:
        raise ValueError(
            "``match_reference_implementation`` is not supported by FLASC."
        )

    # -- Detection and Selection methods
    def __check_selection_method(**kwargs):
        for k, v in kwargs.items():
            if v not in ("eom", "leaf"):
                raise ValueError(
                    f"Invalid ``{k}``: {v}\n"
                    f'Should be one of: "eom", "leaf"\n'
                )

    __check_selection_method(
        branch_selection_method=branch_selection_method,
        cluster_selection_method=cluster_selection_method,
    )
    if branch_detection_method not in ("core", "full"):
        raise ValueError(
            f"Invalid ``branch_detection_method``: {branch_detection_method}\n"
            'Should be one of: "core", "full"\n'
        )

    for bad_arg in [
        "branch_selection_strategy",
        "cluster_selection_strategy",
        "branch_detection_strategy",
    ]:
        if bad_arg in kwargs:
            raise ValueError(
                f"``{bad_arg}`` value ignored! Did you mean to specify "
                f"a value for the ``{bad_arg[:-7]}_method`` parameter?"
            )

    # -- Override clustering
    def __1d_numpy_array_with_shape(
        length=X.shape[0], dtype=np.integer, **kwargs
    ):
        for k, v in kwargs.items():
            if not (
                isinstance(v, np.ndarray)
                and np.issubdtype(v.dtype, dtype)
                and len(v.shape) == 1
                and v.shape[0] == length
            ):
                raise ValueError(
                    f"{k} should be a 1D {dtype} numpy array with "
                    f"{length} points, {v} given."
                )

    if override_cluster_labels is not None:
        if algorithm.startswith("boruvka"):
            raise ValueError(
                "Cannot use Boruvka algorithm with overridden clusters."
            )
        __1d_numpy_array_with_shape(
            override_cluster_labels=override_cluster_labels, dtype=np.integer
        )
        override_cluster_labels = override_cluster_labels.astype(np.intp)

        def __check_override_labels(labels):
            unique_values = np.unique(labels)
            unique_values = unique_values[unique_values >= 0]
            if not np.all(np.diff(unique_values) == 1) or unique_values[0] != 0:
                raise ValueError(
                    "Overriden cluster labels are not consecutive!"
                )

        __check_override_labels(override_cluster_labels)

        if override_cluster_probabilities is None:
            override_cluster_probabilities = np.ones(
                X.shape[0], dtype=np.double
            )
        else:
            __1d_numpy_array_with_shape(
                override_cluster_probabilities=override_cluster_probabilities,
                dtype=np.floating,
            )
            override_cluster_probabilities = (
                override_cluster_probabilities.astype(np.double)
            )
    else:
        if override_cluster_probabilities is not None:
            raise ValueError(
                "Cannot specify ``override_cluster_probabilities`` without"
                " specifying ``override_cluster_labels``."
            )

    # -- Input data
    if metric != "precomputed":
        X = check_array(X, force_all_finite=False)
    else:
        if issparse(X):
            raise ValueError("Sparse distance matrices not supported yet.")
        check_precomputed_distance_matrix(X)
    X = np.ascontiguousarray(X.astype(np.double))
    min_samples = min(X.shape[0] - 1, min_samples)
    if min_samples == 0:
        min_samples = 1

    # -- Extract State
    run_generic = algorithm == "generic" or metric not in FAST_METRICS
    run_override = override_cluster_labels is not None
    run_core = branch_detection_method == "core"

    # -- Memory and num_jobs
    if isinstance(memory, str):
        memory = Memory(memory, verbose=0)

    if num_jobs is not None:
        if not np.issubdtype(type(num_jobs), np.integer):
            raise ValueError("``num_jobs`` should be an integer number.")
        if num_jobs < 1:
            num_jobs = max(cpu_count() + 1 + num_jobs, 1)
        # Create thread pool
        if num_jobs > 1:
            thread_pool = Parallel(n_jobs=num_jobs, max_nbytes=None)
        else:
            thread_pool = SequentialPool()
    else:
        if X.shape[0] > 125000:
            thread_pool = Parallel(n_jobs=4, max_nbytes=None)
        else:
            thread_pool = SequentialPool()

    # - Perform clustering
    # -- Compute single linkage hierarchy
    # Declare variables as None
    space_tree = None  # not used with generic
    reachability = None  # not used with space tree
    single_linkage_tree = None  # not used with override
    min_spanning_tree = None  # not used with override
    core_distances = None
    neighbours = None  # not used with branch_detection_method != 'core'

    # Fill values depending on the cases
    if run_generic:
        if run_override:
            (reachability, core_distances, neighbours) = memory.cache(
                _hdbscan_generic_reachability
            )(
                X,
                min_samples=min_samples,
                alpha=alpha,
                metric=metric,
                **kwargs,
            )
        else:
            (
                single_linkage_tree,
                min_spanning_tree,
                reachability,
                core_distances,
                neighbours,
            ) = memory.cache(_hdbscan_generic_linkage)(
                X,
                min_samples=min_samples,
                alpha=alpha,
                metric=metric,
                **kwargs,
            )
    else:
        space_tree = memory.cache(_hdbscan_space_tree)(
            X, metric=metric, algorithm=algorithm, leaf_size=leaf_size, **kwargs
        )
        if run_override:
            (core_distances, neighbours) = memory.cache(
                _hdbscan_space_tree_core_dists_prims, ignore=["thread_pool"]
            )(space_tree, min_samples=min_samples, thread_pool=thread_pool)
        else:
            (
                single_linkage_tree,
                min_spanning_tree,
                core_distances,
                neighbours,
            ) = memory.cache(
                _hdbscan_space_tree_linkage, ignore=["thread_pool"]
            )(
                space_tree,
                min_samples=min_samples,
                metric=metric,
                alpha=alpha,
                algorithm=algorithm,
                leaf_size=leaf_size,
                approx_min_span_tree=approx_min_span_tree,
                thread_pool=thread_pool,
                **kwargs,
            )

    # -- Perform cluster segmentation
    # Declare variables as None
    cluster_labels = override_cluster_labels
    cluster_probabilities = override_cluster_probabilities
    cluster_persistence = None
    condensed_tree = None

    # Fill values depending on the cases
    if not run_override:
        (
            cluster_labels,
            cluster_probabilities,
            cluster_persistence,
            condensed_tree,
        ) = memory.cache(_hdbscan_extract_clusters)(
            single_linkage_tree,
            min_cluster_size=min_cluster_size,
            cluster_selection_method=cluster_selection_method,
            allow_single_cluster=allow_single_cluster,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_cluster_size=max_cluster_size,
        )

    # - Detect branches
    # -- List the points in each cluster
    if run_override:
        num_clusters = len(np.unique(cluster_labels)) - int(
            -1 in cluster_labels
        )
    else:
        num_clusters = len(cluster_persistence)
    cluster_points = memory.cache(_extract_cluster_points)(
        cluster_labels, num_clusters
    )

    # -- Setup branch threading when num_jobs is not specified
    # No threading benefit for generic and core detection method
    if num_jobs is None and run_core or run_generic:
        thread_pool = SequentialPool()

    # -- Extract each cluster's MST
    if not run_override:
        cluster_spanning_trees = memory.cache(_extract_cluster_msts)(
            min_spanning_tree, cluster_labels, num_clusters
        )
    elif run_generic:
        cluster_spanning_trees = memory.cache(
            _compute_msts_in_cluster_generic, ignore=["thread_pool"]
        )(reachability, cluster_points, thread_pool)
    else:
        cluster_spanning_trees = memory.cache(
            _compute_msts_in_cluster_space_tree, ignore=["thread_pool"]
        )(
            space_tree,
            core_distances,
            cluster_points,
            thread_pool,
            metric=metric,
            alpha=alpha,
            **kwargs,
        )

    # -- Compute branch single-linkage
    (
        cluster_centralities,
        cluster_linkage_trees,
        cluster_approximation_graphs,
    ) = memory.cache(_compute_branch_linkage, ignore=["thread_pool"])(
        space_tree,  # None if run_generic
        reachability,  # None if not run_generic
        core_distances,
        neighbours,  # None if not run_core
        cluster_probabilities,
        cluster_spanning_trees,
        cluster_points,
        thread_pool,
        metric=metric,
        run_core=run_core,
        run_generic=run_generic,
        run_override=run_override,
        **kwargs,
    )

    # -- Extract branch segmentation
    (
        branch_labels,
        branch_probabilities,
        branch_persistences,
        cluster_condensed_trees,
    ) = memory.cache(_compute_branch_segmentation, ignore=["thread_pool"])(
        cluster_linkage_trees,
        thread_pool,
        min_branch_size=min_branch_size,
        allow_single_branch=allow_single_branch,
        branch_selection_method=branch_selection_method,
        branch_selection_persistence=branch_selection_persistence,
        max_branch_size=max_branch_size,
    )

    # - Assign labels and probabilities
    (
        labels,
        probabilities,
        branch_labels,
        branch_probabilities,
        cluster_centralities,
    ) = memory.cache(_update_labelling)(
        cluster_labels,
        cluster_probabilities,
        cluster_points,
        cluster_centralities,
        branch_labels,
        branch_probabilities,
        branch_persistences,
        label_sides_as_branches=label_sides_as_branches,
    )

    return (
        # Combined result
        labels,
        probabilities,
        # Clustering result
        cluster_labels,
        cluster_probabilities,
        cluster_persistence,
        # Branching result
        branch_labels,
        branch_probabilities,
        branch_persistences,
        # Data to clusters
        condensed_tree,
        single_linkage_tree,
        min_spanning_tree,
        # Clusters to branches
        cluster_approximation_graphs,
        cluster_condensed_trees,
        cluster_linkage_trees,
        cluster_centralities,
        cluster_points,
    )


class SequentialPool:
    """API of a Parallel pool but sequential execution"""

    def __init__(self):
        self.n_jobs = 1

    def __call__(self, jobs):
        return [fun(*args, **kwargs) for (fun, args, kwargs) in jobs]
