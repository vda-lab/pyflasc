"""
Tests for FLASC clustering algorithm
Shamelessly based on (i.e. ripped off from) the HDBSCAN test code
"""
import numbers
from functools import wraps
from tempfile import mkdtemp

import numpy as np
import pytest
from scipy import sparse, stats
from scipy.spatial import distance
from scipy.stats import mode
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state, shuffle as util_shuffle
from sklearn.utils._testing import (
    assert_raises,
)

from flasc import FLASC, flasc
from flasc.prediction import (
    approximate_predict,
    branch_centrality_vectors,
    update_labels_with_branch_centrality,
    branch_membership_from_centrality,
)


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed.

    Parameters
    ----------
    func
    """

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib

            matplotlib.use("Agg")
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt

            plt.figure()
        except ImportError:
            pytest.skip("Matplotlib not available.")
        else:
            res = func(*args, **kwargs)
            plt.close("all")
            return res

    return run_test


def if_pandas(func):
    """Test decorator that skips test if pandas not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import pandas
        except ImportError:
            pytest.skip("Pandas not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def if_networkx(func):
    """Test decorator that skips test if networkx not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import networkx
        except ImportError:
            pytest.skip("NetworkX not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def if_pygraphfiz(func):
    """Test decorator that skips test if networkx or pygraphviz is not installed."""

    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import networkx
            import pygraphviz
        except ImportError:
            pytest.skip("NetworkX or pygraphviz not available.")
        else:
            return func(*args, **kwargs)

    return run_test


def make_branches(n_samples=100, shuffle=True, noise=None, random_state=None):
    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 3
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)

    outer_circ_x = np.cos(np.linspace(np.pi / 2, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(np.pi / 2, np.pi, n_samples_out)) - 1
    inner_circ_x = np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in))

    X = np.vstack(
        [
            np.append(outer_circ_x, inner_circ_x),
            np.append(outer_circ_y, inner_circ_y),
        ]
    ).T
    y = np.hstack(
        [
            np.zeros(n_samples_out, dtype=np.intp),
            np.ones(n_samples_in, dtype=np.intp),
        ]
    )

    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y


def generate_noisy_data():
    blobs, yBlobs = make_blobs(
        n_samples=50,
        centers=[(-0.75, 2.25), (2.0, -0.5)],
        cluster_std=0.2,
        random_state=3,
    )
    moons, _ = make_branches(n_samples=150, noise=0.06, random_state=3)
    yMoons = np.full(moons.shape[0], 2)
    np.random.seed(5)
    noise = np.random.uniform(-1.0, 3.0, (50, 2))
    yNoise = np.full(50, -1)
    return (
        np.vstack((blobs, moons, noise)),
        np.concatenate((yBlobs, yMoons, yNoise)),
    )


def homogeneity(labels1, labels2):
    num_missed = 0.0
    for label in set(labels1):
        matches = labels2[labels1 == label]
        match_mode = mode(matches, keepdims=True)[0][0]
        num_missed += np.sum(matches != match_mode)

    for label in set(labels2):
        matches = labels1[labels2 == label]
        match_mode = mode(matches, keepdims=True)[0][0]
        num_missed += np.sum(matches != match_mode)

    return num_missed / 2.0


def check_num_branches_and_clusters(C, n_branches=6, n_clusters=3):
    """Checks number of detected branches and clusters equal to expected values."""
    if isinstance(C, FLASC):
        # C is an FLASC clusterer object
        n_detected_branches = len(np.unique(C.labels_)) - int(-1 in C.labels_)
        n_detected_clusters = len(np.unique(C.cluster_labels_)) - int(
            -1 in C.cluster_labels_
        )
    else:
        # C is the output of flasc function.
        n_detected_branches = len(np.unique(C[0])) - int(-1 in C[0])
        n_detected_clusters = len(np.unique(C[2])) - int(-1 in C[2])
    # Check that values are correct
    assert n_detected_branches == n_branches
    assert n_detected_clusters == n_clusters
    return C


X, y = generate_noisy_data()
X = StandardScaler().fit_transform(X)

X_missing_data = X.copy()
X_missing_data[0] = [np.nan, 1]
X_missing_data[6] = [np.nan, np.nan]


# --- Input types


def test_flasc_sparse_distance_matrix():
    """Sparse inputs not supported!"""
    D = distance.squareform(distance.pdist(X))
    D /= np.max(D)
    threshold = stats.scoreatpercentile(D.flatten(), 80)
    D[D >= threshold] = 0.0
    D = sparse.csr_matrix(D)
    D.eliminate_zeros()
    assert_raises(ValueError, flasc, D, metric="precomputed")


def test_missing_data():
    """Tests if nan data are treated as infinite distance from all other points
    and assigned to -1 cluster"""

    def _asserts():
        assert model.labels_[0] == -1
        assert model.labels_[6] == -1
        assert model.cluster_labels_[0] == -1
        assert model.cluster_labels_[6] == -1
        assert model.probabilities_[0] == 0
        assert model.probabilities_[6] == 0
        assert model.cluster_probabilities_[0] == 0
        assert model.cluster_probabilities_[6] == 0
        assert model.branch_probabilities_[0] == 0
        assert model.branch_probabilities_[6] == 0
        for tree in model._cluster_approximation_graphs:
            assert np.all(tree[:, 0] != 0)
            assert np.all(tree[:, 1] != 0)
            assert np.all(tree[:, 0] != 6)
            assert np.all(tree[:, 1] != 6)
        for pts in model.cluster_points_:
            assert np.all(pts != 0)
            assert np.all(pts != 6)
        assert np.allclose(clean_model.labels_, model.labels_[clean_indices])

    clean_indices = list(range(1, 6)) + list(range(7, X.shape[0]))
    model = FLASC().fit(X_missing_data)
    clean_model = FLASC().fit(X_missing_data[clean_indices])
    _asserts()
    clean_indices = list(range(1, 6)) + list(range(7, X.shape[0]))
    model = FLASC(override_cluster_labels=y).fit(X_missing_data)
    clean_model = FLASC(override_cluster_labels=y[clean_indices]).fit(
        X_missing_data[clean_indices]
    )
    _asserts()


def test_flasc_distance_matrix():
    """Tests branch detection method + override combinations on precomputed
    distance matrix."""
    D = distance.squareform(distance.pdist(X))
    for detection_method in ["core", "full"]:
        check_num_branches_and_clusters(
            flasc(
                D,
                metric="precomputed",
                branch_detection_method=detection_method,
            )
        )
        res = check_num_branches_and_clusters(
            flasc(
                D,
                metric="precomputed",
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            )
        )
        assert res[8] is None
        assert res[9] is None
        assert res[10] is None
        check_num_branches_and_clusters(
            FLASC(
                metric="precomputed",
                branch_detection_method=detection_method,
            ).fit(D)
        )
        check_num_branches_and_clusters(
            FLASC(
                metric="precomputed",
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            ).fit(D)
        )


def test_flasc_feature_vector():
    """Tests branch detection method + override combinations on feature vector
    inputs."""
    for detection_method in ["core", "full"]:
        check_num_branches_and_clusters(
            flasc(
                X,
                branch_detection_method=detection_method,
            )
        )
        res = check_num_branches_and_clusters(
            flasc(
                X,
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            )
        )
        assert res[8] is None
        assert res[9] is None
        assert res[10] is None
        check_num_branches_and_clusters(
            FLASC(
                branch_detection_method=detection_method,
            ).fit(X)
        )
        check_num_branches_and_clusters(
            FLASC(
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            ).fit(X)
        )


def test_flasc_high_dimensional():
    """Tests branch detection method + override combinations on high dimensional
    feature vector inputs."""
    H, y = make_blobs(n_samples=50, random_state=0, n_features=64)
    H = StandardScaler().fit_transform(H)
    for detection_method in ["core", "full"]:
        check_num_branches_and_clusters(
            flasc(
                H,
                branch_detection_method=detection_method,
            ),
            n_branches=3,
        )
        res = check_num_branches_and_clusters(
            flasc(
                H,
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            ),
            n_branches=3,
        )
        assert res[8] is None
        assert res[9] is None
        assert res[10] is None
        check_num_branches_and_clusters(
            FLASC(
                branch_detection_method=detection_method,
            ).fit(H),
            n_branches=3,
        )
        check_num_branches_and_clusters(
            FLASC(
                override_cluster_labels=y,
                branch_detection_method=detection_method,
            ).fit(H),
            n_branches=3,
        )


def test_flasc_input_lists():
    X = [[1.0, 2.0], [3.0, 4.0]]
    FLASC().fit(X)  # must not raise exception


def test_flasc_sparse():
    sparse_X = sparse.csr_matrix(X)
    assert_raises(TypeError, FLASC().fit, sparse_X)


# --- MST construction algorithms


def test_flasc_mst_algorithms():
    for algorithm in [
        "generic",
        "prims_kdtree",
        "prims_balltree",
        "boruvka_kdtree",
        "boruvka_balltree",
    ]:
        for detection_method in ["core", "full"]:

            check_num_branches_and_clusters(
                flasc(
                    X,
                    algorithm=algorithm,
                    min_branch_size=10,
                    branch_detection_method=detection_method,
                )
            )
            check_num_branches_and_clusters(
                FLASC(
                    algorithm=algorithm,
                    min_branch_size=10,
                    branch_detection_method=detection_method,
                ).fit(X)
            )
    for algorithm in ["generic", "prims_kdtree", "prims_balltree"]:
        for detection_method in ["core", "full"]:
            check_num_branches_and_clusters(
                flasc(
                    X,
                    algorithm=algorithm,
                    min_branch_size=10,
                    override_cluster_labels=y,
                    branch_detection_method=detection_method,
                )
            )
            check_num_branches_and_clusters(
                FLASC(
                    algorithm=algorithm,
                    min_branch_size=10,
                    override_cluster_labels=y,
                    branch_detection_method=detection_method,
                ).fit(X)
            )
    assert_raises(ValueError, flasc, X, algorithm="prims_kdtree", metric="russelrao")
    assert_raises(
        ValueError, flasc, X, algorithm="boruvka_kdtree", metric="russelrao"
    )
    assert_raises(ValueError, flasc, X, algorithm="prims_balltree", metric="cosine")
    assert_raises(
        ValueError, flasc, X, algorithm="boruvka_balltree", metric="cosine"
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        algorithm="boruvka_kdtree",
        override_cluster_labels=y,
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        algorithm="boruvka_balltree",
        override_cluster_labels=y,
    )


def test_flasc_best_balltree_metric():
    check_num_branches_and_clusters(
        flasc(X, metric="seuclidean", V=np.ones(X.shape[1]))
    )
    check_num_branches_and_clusters(
        FLASC(metric="seuclidean", V=np.ones(X.shape[1])).fit(X)
    )


def test_flasc_kdtree_matches():
    """
    Detects issues with minimum spanning tree construction. Don't disregard
    higher homogeneity values as insignificant!
    """
    res = flasc(X, algorithm="generic", branch_detection_method="core")
    labels_generic = res[2]
    res = flasc(X, algorithm="prims_kdtree", branch_detection_method="core")
    labels_prims = res[2]
    res = flasc(X, algorithm="boruvka_kdtree", branch_detection_method="core")
    labels_boruvka = res[2]
    assert homogeneity(labels_generic, labels_prims) <= 6
    assert homogeneity(labels_generic, labels_boruvka) <= 6
    assert homogeneity(labels_prims, labels_boruvka) <= 6


def test_flasc_balltree_matches():
    """
    Detects issues with minimum spanning tree construction. Don't disregard
    higher homogeneity values as insignificant!
    """
    res = flasc(X, algorithm="generic", branch_detection_method="core")
    labels_generic = res[2]
    res = flasc(X, algorithm="prims_balltree", branch_detection_method="core")
    labels_prims = res[2]
    res = flasc(X, algorithm="boruvka_balltree", branch_detection_method="core")
    labels_boruvka = res[2]
    assert homogeneity(labels_generic, labels_prims) <= 6
    assert homogeneity(labels_generic, labels_boruvka) <= 6
    assert homogeneity(labels_prims, labels_boruvka) <= 6


# --- Varying min cluster sizes


def test_flasc_no_clusters():
    check_num_branches_and_clusters(
        flasc(X, min_cluster_size=len(X) + 1), n_branches=0, n_clusters=0
    )
    check_num_branches_and_clusters(
        FLASC(min_cluster_size=len(X) + 1).fit(X), n_branches=0, n_clusters=0
    )


def test_flasc_min_cluster_size():
    for min_cluster_size in range(2, len(X) + 1, 1):
        res = flasc(X, min_cluster_size=min_cluster_size)
        labels = res[0]
        cluster_labels = res[2]
        true_labels = [label for label in labels if label != -1]
        true_cluster_labels = [label for label in cluster_labels if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size
        if len(true_cluster_labels) != 0:
            assert np.min(np.bincount(true_cluster_labels)) >= min_cluster_size

        c = FLASC(min_cluster_size=min_cluster_size).fit(X)
        true_labels = [label for label in c.labels_ if label != -1]
        true_cluster_labels = [label for label in c.cluster_labels_ if label != -1]
        if len(true_labels) != 0:
            assert np.min(np.bincount(true_labels)) >= min_cluster_size
        if len(true_cluster_labels) != 0:
            assert np.min(np.bincount(true_cluster_labels)) >= min_cluster_size


# --- Metrics


def test_flasc_callable_metric():
    # metric is the function reference, not the string key.
    metric = distance.euclidean
    check_num_branches_and_clusters(flasc(X, metric=metric))
    check_num_branches_and_clusters(FLASC(metric=metric).fit(X))


# --- Plotting objects


def test_approximation_graph_plot():
    clusterer = FLASC().fit(X)
    g = clusterer.cluster_approximation_graph_
    if_matplotlib(g.plot)(positions=X)
    if_pygraphfiz(if_matplotlib(g.plot))(node_color="x", feature_names=["x", "y"])
    if_pygraphfiz(if_matplotlib(g.plot))(edge_color="centrality", node_alpha=0)
    if_pygraphfiz(if_matplotlib(g.plot))(node_color=X[:, 0], node_alpha=0)
    if_pygraphfiz(if_matplotlib(g.plot))(
        edge_color=g._edges["centrality"], node_alpha=0
    )


def test_condensed_tree_plot():
    clusterer = FLASC().fit(X)
    if_matplotlib(clusterer.condensed_tree_.plot)(
        select_clusters=True,
        label_clusters=True,
        selection_palette=("r", "g", "b"),
        cmap="Reds",
    )
    if_matplotlib(clusterer.condensed_tree_.plot)(
        log_size=True, colorbar=False, cmap="none"
    )


def test_single_linkage_tree_plot():
    clusterer = FLASC().fit(X)
    if_matplotlib(clusterer.single_linkage_tree_.plot)(cmap="Reds")
    if_matplotlib(clusterer.single_linkage_tree_.plot)(
        vary_line_width=False,
        truncate_mode="lastp",
        p=10,
        cmap="none",
        colorbar=False,
    )


def test_min_span_tree_plot():
    clusterer = FLASC().fit(X)
    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(edge_cmap="Reds")

    H, y = make_blobs(n_samples=50, random_state=0, n_features=10)
    H = StandardScaler().fit_transform(H)

    clusterer = FLASC().fit(H)
    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(
        edge_cmap="Reds", vary_line_width=False, colorbar=False
    )

    H, y = make_blobs(n_samples=50, random_state=0, n_features=40)
    H = StandardScaler().fit_transform(H)

    clusterer = FLASC().fit(H)
    if_matplotlib(clusterer.minimum_spanning_tree_.plot)(
        edge_cmap="Reds", vary_line_width=False, colorbar=False
    )


def test_cluster_condensed_trees_plot():
    clusterer = FLASC().fit(X)
    for t in clusterer.cluster_condensed_trees_:
        if_matplotlib(t.plot)(
            select_clusters=True,
            label_clusters=True,
            selection_palette=("r", "g", "b"),
            cmap="Reds",
        )
        if_matplotlib(t.plot)(log_size=True, colorbar=False, cmap="none")


def test_cluster_single_linkage_tree_plot():
    clusterer = FLASC().fit(X)
    for t in clusterer.cluster_linkage_trees_:
        if_matplotlib(t.plot)(cmap="Reds")
        if_matplotlib(t.plot)(
            vary_line_width=False,
            truncate_mode="lastp",
            p=10,
            cmap="none",
            colorbar=False,
        )


def test_tree_numpy_output_formats():
    clusterer = FLASC().fit(X)
    clusterer.single_linkage_tree_.to_numpy()
    clusterer.condensed_tree_.to_numpy()
    clusterer.minimum_spanning_tree_.to_numpy()
    clusterer.cluster_approximation_graph_.to_numpy()
    for t in clusterer.cluster_condensed_trees_:
        t.to_numpy()
    for t in clusterer.cluster_linkage_trees_:
        t.to_numpy()


def test_tree_pandas_output_formats():
    clusterer = FLASC().fit(X)
    if_pandas(clusterer.condensed_tree_.to_pandas)()
    if_pandas(clusterer.single_linkage_tree_.to_pandas)()
    if_pandas(clusterer.minimum_spanning_tree_.to_pandas)()
    if_pandas(clusterer.cluster_approximation_graph_.to_pandas)()
    for t in clusterer.cluster_condensed_trees_:
        if_pandas(t.to_pandas)()
    for t in clusterer.cluster_linkage_trees_:
        if_pandas(t.to_pandas)()


def test_tree_networkx_output_formats():
    clusterer = FLASC().fit(X)
    if_networkx(clusterer.condensed_tree_.to_networkx)()
    if_networkx(clusterer.single_linkage_tree_.to_networkx)()
    if_networkx(clusterer.minimum_spanning_tree_.to_networkx)()
    if_networkx(clusterer.cluster_approximation_graph_.to_networkx)()
    for t in clusterer.cluster_condensed_trees_:
        if_networkx(t.to_networkx)()
    for t in clusterer.cluster_linkage_trees_:
        if_networkx(t.to_networkx)()


# --- Sklearn Object attributes


def test_hdbscan_object():
    clusterer = FLASC().fit(X)
    scores = clusterer.hdbscan_.outlier_scores_
    assert scores is not None
    labels = clusterer.hdbscan_.dbscan_clustering(0.3)
    n_detected_clusters = len(set(labels)) - int(-1 in labels)
    assert n_detected_clusters == 3


def test_flasc_unavailable_attributes():
    clusterer = FLASC()
    assert_raises(AttributeError, lambda: clusterer.condensed_tree_)
    assert_raises(AttributeError, lambda: clusterer.single_linkage_tree_)
    assert_raises(AttributeError, lambda: clusterer.minimum_spanning_tree_)
    assert_raises(AttributeError, lambda: clusterer.cluster_approximation_graph_)
    assert_raises(AttributeError, lambda: clusterer.cluster_condensed_trees_)
    assert_raises(AttributeError, lambda: clusterer.cluster_linkage_trees_)
    assert_raises(AttributeError, lambda: clusterer.branch_exemplars_)
    assert_raises(AttributeError, lambda: clusterer.cluster_exemplars_)
    assert_raises(AttributeError, lambda: clusterer.relative_validity_)
    assert_raises(AttributeError, lambda: clusterer.hdbscan_)
    assert_raises(
        AttributeError,
        lambda: approximate_predict(clusterer, np.array([[-0.8, 0.0]])),
    )
    # Not available with override clusters
    clusterer = FLASC(override_cluster_labels=y).fit(X)
    assert_raises(AttributeError, lambda: clusterer.condensed_tree_)
    assert_raises(AttributeError, lambda: clusterer.single_linkage_tree_)
    assert_raises(AttributeError, lambda: clusterer.minimum_spanning_tree_)
    clusterer.cluster_approximation_graph_
    clusterer.cluster_condensed_trees_
    clusterer.cluster_linkage_trees_
    clusterer.branch_exemplars_
    assert_raises(AttributeError, lambda: clusterer.cluster_exemplars_)
    assert_raises(AttributeError, lambda: clusterer.relative_validity_)
    assert_raises(AttributeError, lambda: clusterer.hdbscan_)
    assert_raises(
        AttributeError,
        lambda: approximate_predict(clusterer, np.array([[-0.8, 0.0]])),
    )
    # Not available with precomputed distances
    D = distance.squareform(distance.pdist(X))
    clusterer = FLASC(metric="precomputed").fit(D)
    clusterer.condensed_tree_
    assert_raises(AttributeError, lambda: clusterer.minimum_spanning_tree_)
    clusterer.cluster_approximation_graph_
    clusterer.cluster_condensed_trees_
    clusterer.cluster_linkage_trees_
    assert_raises(AttributeError, lambda: clusterer.branch_exemplars_)
    assert_raises(AttributeError, lambda: clusterer.cluster_exemplars_)
    assert_raises(AttributeError, lambda: clusterer.relative_validity_)
    clusterer.hdbscan_
    assert_raises(
        AttributeError,
        lambda: approximate_predict(clusterer, np.array([[-0.8, 0.0]])),
    )


def test_flasc_exemplars():
    clusterer = FLASC().fit(X)
    branch_exemplars = clusterer.branch_exemplars_
    assert branch_exemplars[0] is None
    assert branch_exemplars[1] is None
    assert len(branch_exemplars[2]) == 3
    assert len(clusterer.cluster_exemplars_) == 3


def test_flasc_centroids_medoids():
    branch_centers = np.asarray(
        [[-0.9, -1.0], [-0.9, 0.1], [-0.8, 1.9], [-0.5, 0.0], [1.7, -0.9]]
    )
    cluster_centers = [(-0.86, 1.85), (-0.38, -0.35), (1.57, -1.00)]

    clusterer = FLASC().fit(X)

    centroids = np.asarray([clusterer.weighted_centroid(i) for i in range(5)])
    rounded = np.around(np.asarray(centroids), decimals=1)
    corder = np.lexsort((rounded[:, 1], rounded[:, 0]))
    np.all(np.abs(centroids[corder, :] - branch_centers) < 0.1)

    medoids = np.asarray([clusterer.weighted_medoid(i) for i in range(5)])
    rounded = np.around(np.asarray(medoids), decimals=1)
    corder = np.lexsort((rounded[:, 1], rounded[:, 0]))
    np.all(np.abs(medoids[corder, :] - branch_centers) < 0.1)

    centroids = np.asarray([clusterer.weighted_cluster_centroid(i) for i in range(3)])
    rounded = np.around(np.asarray(centroids), decimals=1)
    corder = np.lexsort((rounded[:, 1], rounded[:, 0]))
    np.all(np.abs(centroids[corder, :] - cluster_centers) < 0.1)

    medoids = np.asarray([clusterer.weighted_cluster_medoid(i) for i in range(3)])
    rounded = np.around(np.asarray(medoids), decimals=1)
    corder = np.lexsort((rounded[:, 1], rounded[:, 0]))
    np.all(np.abs(medoids[corder, :] - cluster_centers) < 0.1)


def test_flasc_no_centroid_medoid_for_noise():
    clusterer = FLASC().fit(X)
    assert_raises(ValueError, clusterer.weighted_centroid, -1)
    assert_raises(ValueError, clusterer.weighted_medoid, -1)
    assert_raises(ValueError, clusterer.weighted_cluster_centroid, -1)
    assert_raises(ValueError, clusterer.weighted_cluster_medoid, -1)


# --- Prediction


def test_flasc_approximate_predict():
    clusterer = FLASC().fit(X)

    # A point on a branch (not noise) exact labels change per run
    l, p, cl, cp, bl, bp = approximate_predict(clusterer, np.array([[-0.8, 0.0]]))
    assert cl[0] > -1
    assert len(clusterer.branch_persistences_[cl[0]]) > 2

    # A point in a cluster
    l, p, cl, cp, bl, bp = approximate_predict(clusterer, np.array([[-0.8, 2.0]]))
    assert l[0] == cl[0]
    assert bl[0] == 0
    assert bp[0] == 1.0

    # A noise point
    l, p, cl, cp, bl, bp = approximate_predict(clusterer, np.array([[1, 3.0]]))
    assert l[0] == -1
    assert cl[0] == -1
    assert cp[0] == 0
    assert p[0] == 0.0
    assert cp[0] == 0.0
    assert bp[0] == 1.0


def test_flasc_weighted_membership():
    clusterer = FLASC().fit(X)

    # Distance to branch roots
    branch_centralities = branch_centrality_vectors(clusterer)
    assert len(branch_centralities) == 3
    assert branch_centralities[0] is None
    assert branch_centralities[1] is None
    assert branch_centralities[2] is not None
    assert branch_centralities[2].shape[1] == 3

    # Update branch labels to closest branch roots
    labels, branch_labels = update_labels_with_branch_centrality(
        clusterer, branch_centralities
    )
    assert np.allclose(
        clusterer.labels_[clusterer.cluster_labels_ != 2],
        labels[clusterer.cluster_labels_ != 2],
    )
    assert np.allclose(
        clusterer.branch_labels_[clusterer.cluster_labels_ != 2],
        branch_labels[clusterer.cluster_labels_ != 2],
    )
    assert ~np.allclose(
        clusterer.labels_[clusterer.cluster_labels_ == 2],
        labels[clusterer.cluster_labels_ == 2],
    )
    assert ~np.allclose(
        clusterer.branch_labels_[clusterer.cluster_labels_ == 2],
        branch_labels[clusterer.cluster_labels_ == 2],
    )

    # Softmax to make centralities act as probability
    branch_memberships = branch_membership_from_centrality(branch_centralities)
    assert branch_memberships[0] is None
    assert branch_memberships[1] is None
    assert branch_memberships[2] is not None
    assert branch_memberships[2].shape[1] == 3
    assert np.allclose(np.sum(branch_memberships[2], axis=1), 1.0)


# --- Input arguments


def test_flasc_badargs():
    D = distance.squareform(distance.pdist(X))
    assert_raises(AttributeError, flasc, X="fail")
    assert_raises(AttributeError, flasc, X=None)
    assert_raises(ValueError, flasc, X, min_cluster_size=-1)
    assert_raises(ValueError, flasc, X, min_cluster_size=0)
    assert_raises(ValueError, flasc, X, min_cluster_size=1)
    assert_raises(ValueError, flasc, X, min_cluster_size=2.0)
    assert_raises(ValueError, flasc, X, min_cluster_size="fail")
    assert_raises(ValueError, flasc, X, min_branch_size=-1)
    assert_raises(ValueError, flasc, X, min_branch_size=0)
    assert_raises(ValueError, flasc, X, min_branch_size=1)
    assert_raises(ValueError, flasc, X, min_branch_size=2.0)
    assert_raises(ValueError, flasc, X, min_branch_size="fail")
    assert_raises(ValueError, flasc, X, min_samples=-1)
    assert_raises(ValueError, flasc, X, min_samples=0)
    assert_raises(ValueError, flasc, X, min_samples=1.0)
    assert_raises(ValueError, flasc, X, min_samples="fail")
    assert_raises(ValueError, flasc, X, num_jobs="fail")
    assert_raises(ValueError, flasc, X, num_jobs=1.5)
    assert_raises(ValueError, flasc, X, metric=None)
    assert_raises(ValueError, flasc, X, metric="imperial")
    assert_raises(ValueError, flasc, X, metric="minkowski", p=-1)
    assert_raises(ValueError, flasc, X, metric="minkowski", p=-0.1)
    assert_raises(TypeError, flasc, X, metric="minkowski", p="fail")
    assert_raises(TypeError, flasc, X, metric="minkowski", p=None)
    assert_raises(ValueError, flasc, X, alpha=-1)
    assert_raises(ValueError, flasc, X, alpha=-0.1)
    assert_raises(ValueError, flasc, X, alpha="fail")
    assert_raises(ValueError, flasc, X, leaf_size=0)
    assert_raises(ValueError, flasc, X, leaf_size=1.0)
    assert_raises(ValueError, flasc, X, leaf_size="fail")
    assert_raises(ValueError, flasc, X, cluster_selection_epsilon=-1)
    assert_raises(ValueError, flasc, X, cluster_selection_epsilon=-0.1)
    assert_raises(ValueError, flasc, X, branch_selection_persistence=-1)
    assert_raises(ValueError, flasc, X, branch_selection_persistence=-0.1)
    assert_raises(
        ValueError,
        flasc,
        X,
        metric="precomputed",
        algorithm="boruvka_kdtree",
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        metric="precomputed",
        algorithm="boruvka_balltree",
    )
    assert_raises(
        ValueError, flasc, X, metric="precomputed", algorithm="prims_kdtree"
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        metric="precomputed",
        algorithm="prims_balltree",
    )
    assert_raises(ValueError, flasc, X, branch_selection_strategy="leaf")
    assert_raises(
        ValueError,
        flasc,
        D,
        metric="precomputed",
        branch_selection_strategy="leaf",
    )
    assert_raises(ValueError, flasc, X, cluster_selection_strategy="leaf")
    assert_raises(
        ValueError,
        flasc,
        D,
        metric="precomputed",
        cluster_selection_strategy="leaf",
    )
    assert_raises(ValueError, flasc, X, branch_detection_strategy="full")
    assert_raises(
        ValueError,
        flasc,
        D,
        metric="precomputed",
        branch_detection_strategy="full",
    )
    assert_raises(Exception, flasc, X, algorithm="something_else")
    assert_raises(ValueError, flasc, X, cluster_selection_method="something_else")
    assert_raises(ValueError, flasc, X, branch_selection_method="something_else")
    assert_raises(ValueError, flasc, X, branch_detection_method="something_else")
    assert_raises(ValueError, flasc, X, override_cluster_labels=0.0)
    assert_raises(ValueError, flasc, X, override_cluster_labels=[])
    assert_raises(ValueError, flasc, X, override_cluster_labels=np.asarray([0.0]))
    assert_raises(
        ValueError,
        flasc,
        X,
        override_cluster_labels=y,
        override_cluster_probabilities=[],
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        override_cluster_labels=y,
        override_cluster_probabilities=np.asarray([]),
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        override_cluster_probabilities=np.ones(X.shape[0]),
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        override_cluster_labels=y,
        algorithm="boruvka_kdtree",
    )
    assert_raises(
        ValueError,
        flasc,
        X,
        override_cluster_labels=y,
        algorithm="boruvka_balltree",
    )


# --- Caching


def test_flasc_caching():
    cachedir = mkdtemp()
    c1 = FLASC(memory=cachedir, min_samples=6).fit(X)
    c2 = FLASC(memory=cachedir, min_samples=6, min_cluster_size=6).fit(X)
    n_branches1 = len(set(c1.labels_)) - int(-1 in c1.labels_)
    n_branches2 = len(set(c2.labels_)) - int(-1 in c2.labels_)
    assert n_branches1 == n_branches2
    n_clusters1 = len(set(c1.cluster_labels_)) - int(-1 in c1.cluster_labels_)
    n_clusters2 = len(set(c2.cluster_labels_)) - int(-1 in c2.cluster_labels_)
    assert n_clusters1 == n_clusters2


# --- Input parameters


def test_flasc_allow_single_cluster_with_epsilon():
    np.random.seed(0)
    no_structure = np.random.rand(150, 2)
    # without epsilon we should see no noise points as children of root.
    c = FLASC(
        min_cluster_size=5,
        cluster_selection_epsilon=0.0,
        cluster_selection_method="eom",
        allow_single_cluster=True,
    ).fit(no_structure)
    assert np.all(c.cluster_labels_ == 0)

    # for this random seed an epsilon of 0.2 will produce exactly 2 noise
    # points at that cut in single linkage.
    c = FLASC(
        min_cluster_size=5,
        cluster_selection_epsilon=0.2,
        cluster_selection_method="eom",
        allow_single_cluster=True,
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.cluster_labels_, return_counts=True)
    assert len(unique_labels) == 2
    assert counts[unique_labels == -1] == 2


def test_flasc_allow_single_branch_with_persistence():
    np.random.seed(0)
    no_structure = np.random.rand(150, 2)
    no_structure_labels = np.zeros(no_structure.shape[0], dtype=np.intp)

    # Without persistence, find 4 branches
    c = FLASC(
        min_cluster_size=5,
        override_cluster_labels=no_structure_labels,
        branch_detection_method='core',
        branch_selection_method='leaf',
        allow_single_branch=True,
        branch_selection_persistence=0,
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 5
    assert np.sum(c.branch_probabilities_ == 0) == 84

    # At persistence 1, num prob == 0 decreases to 67
    c = FLASC(
        min_cluster_size=5,
        override_cluster_labels=no_structure_labels,
        branch_detection_method='core',
        branch_selection_method='leaf',
        allow_single_branch=True,
        branch_selection_persistence=1,
    ).fit(no_structure)
    unique_labels, counts = np.unique(c.labels_, return_counts=True)
    assert len(unique_labels) == 1
    assert np.sum(c.branch_probabilities_ == 0) == 0


def test_flasc_label_sides_as_branches():
    check_num_branches_and_clusters(
        flasc(
            X,
            min_samples=3,
            min_cluster_size=10,
            label_sides_as_branches=True,
            branch_detection_method="core",
        ),
        n_branches=6,
    )
    check_num_branches_and_clusters(
        FLASC(
            min_samples=3,
            min_cluster_size=10,
            label_sides_as_branches=True,
            branch_detection_method="core",
        ).fit(X),
        n_branches=6,
    )


def test_flasc_override_clusters_with_probability():
    res = check_num_branches_and_clusters(
        flasc(
            X,
            min_branch_size=10,
            override_cluster_labels=y,
            override_cluster_probabilities=np.ones(X.shape[0]) / 2,
        )
    )
    assert res[8] is None
    assert res[9] is None
    assert res[10] is None

    check_num_branches_and_clusters(
        FLASC(
            override_cluster_labels=y,
            override_cluster_probabilities=np.ones(X.shape[0]),
        ).fit(X)
    )


def test_flasc_leaf_methods():
    check_num_branches_and_clusters(
        flasc(
            X,
            cluster_selection_method="leaf",
            branch_selection_method="leaf",
        ),
        n_branches=10,
        n_clusters=10,
    )
    check_num_branches_and_clusters(
        FLASC(
            cluster_selection_method="leaf",
            branch_selection_method="leaf",
        ).fit(X),
        n_branches=10,
        n_clusters=10,
    )
