# Plotting objects for FLASC
# Author: Leland McInnes <leland.mcinnes@gmail.com>
# Adapted for FLASC by Jelmer Bot
# License: BSD 3 clause
import numpy as np
from warnings import warn
from hdbscan.plots import _bfs_from_cluster_tree, _get_leaves
from hdbscan._hdbscan_tree import compute_stability, epsilon_search

CB_LEFT = 0
CB_RIGHT = 1
CB_BOTTOM = 2
CB_TOP = 3


class ApproximationGraph:
    """
    The approximation graph structure describing the connectivity within clusters
    used to detect branches.

    Parameters
    ----------
    approximation_graphs : list[np.ndarray], shape (n_clusters),

    labels : np.ndarray, shape (n_samples, )
        FLASC labelling for each point.

    probabilities : np.ndarray, shape (n_samples, )
        FLASC probabilities for each point.

    cluster_labels : np.ndarray, shape (n_samples, )
        HDBSCAN* labelling for each point.

    cluster_probabilities : np.ndarray, shape (n_samples, )
        HDBSCAN* probabilities for each point.

    cluster_centralities : np.ndarray, shape (n_samples, )
        Centrality values for each point in a cluster.

    branch_labels : np.ndarray, shape (n_samples, )
        Within cluster branch labels for each point.

    branch_probabilities : np.ndarray, shape (n_samples, )
        Within cluster branch membership strengths for each point.

    Attributes
    ----------
    point_mask : np.ndarray[bool], shape (n_samples)
        A mask to extract points within clusters from the raw data.
    """

    def __init__(
        self,
        approximation_graphs,
        labels,
        probabilities,
        cluster_labels,
        cluster_probabilities,
        cluster_centralities,
        branch_labels,
        branch_probabilities,
        raw_data=None,
    ):
        self._edges = np.array(
            [
                (edge[0], edge[1], edge[2], edge[3], cluster)
                for cluster, edges in enumerate(approximation_graphs)
                for edge in edges
            ],
            dtype=[
                ("parent", np.intp),
                ("child", np.intp),
                ("centrality", np.float64),
                ("mutual_reachability", np.float64),
                ("cluster", np.intp),
            ],
        )
        self.point_mask = cluster_labels >= 0
        self._raw_data = raw_data[self.point_mask, :] if raw_data is not None else None
        self._points = np.array(
            [
                (
                    i,
                    labels[i],
                    probabilities[i],
                    cluster_labels[i],
                    cluster_probabilities[i],
                    cluster_centralities[i],
                    branch_labels[i],
                    branch_probabilities[i],
                )
                for i in np.where(self.point_mask)[0]
            ],
            dtype=[
                ("id", np.intp),
                ("label", np.intp),
                ("probability", np.float64),
                ("cluster_label", np.intp),
                ("cluster_probability", np.float64),
                ("cluster_centrality", np.float64),
                ("branch_label", np.intp),
                ("branch_probability", np.float64),
            ],
        )
        self._pos = None

    def plot(
        self,
        positions=None,
        feature_names=None,
        node_color="label",
        node_vmin=None,
        node_vmax=None,
        node_cmap="viridis",
        node_alpha=1,
        # node_desat=None,
        node_size=1,
        node_marker="o",
        edge_color="k",
        edge_vmin=None,
        edge_vmax=None,
        edge_cmap="viridis",
        edge_alpha=1,
        edge_width=1,
    ):
        """
        Plots the Approximation graph, requires networkx and matplotlib.

        Parameters
        ----------
        positions : np.ndarray, shape (n_samples, 2) (default = None)
            A position for each data point in the graph or each data point in the
            raw data. When None, the function attempts to compute graphviz'
            sfdp layout, which requires pygraphviz to be installed and available.

        node_color : str (default = 'label')
            The point attribute to to color the nodes by. Possible values:
            - id
            - label
            - probability
            - cluster_label
            - cluster_probability
            - cluster_centrality
            - branch_label
            - branch_probability,
            - The input data's feature (if available) names if
            ``feature_names`` is specified or ``feature_x`` for the x-th feature
            if no ``feature_names`` are given, or anything matplotlib scatter
            interprets as a color.

        node_vmin : float, (default = None)
            The minimum value to use for normalizing node colors.

        node_vmax : float, (default = None)
            The maximum value to use for normalizing node colors.

        node_cmap : str, (default = 'tab10')
            The cmap to use for coloring nodes.

        node_alpha : float, (default = 1)
            The node transparency value.

        node_size : float, (default = 5)
            The node marker size value.

        node_marker : str, (default = 'o')
            The node marker string.

        edge_color : str (default = 'label')
            The point attribute to to color the nodes by. Possible values:
            - weight
            - mutual reachability
            - centrality,
            - cluster,
            or anything matplotlib linecollection interprets as color.

        edge_vmin : float, (default = None)
            The minimum value to use for normalizing edge colors.

        edge_vmax : float, (default = None)
            The maximum value to use for normalizing edge colors.

        edge_cmap : str, (default = viridis)
            The cmap to use for coloring edges.

        edge_alpha : float, (default = 1)
            The edge transparency value.

        edge_width : float, (default = 1)
            The edge line width size value.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.collections as mc
        except ImportError:
            raise ImportError(
                "You must install the matplotlib library to plot the Approximation Graph."
            )

        # Extract node color data
        if node_color is None:
            pass
        elif isinstance(node_color, str):
            if node_color in self._points.dtype.names:
                if "label" in node_color:
                    node_vmax = 9
                    node_vmin = 0
                    node_cmap = "tab10"
                    node_color = self._points[node_color] % 10
                else:
                    node_color = self._points[node_color]
            elif (
                self._raw_data is not None
                and feature_names is not None
                and node_color in feature_names
            ):
                idx = feature_names.index(node_color)
                node_color = self._raw_data[:, idx]
            elif self._raw_data is not None and node_color.startswith("feature_"):
                idx = int(node_color[8:])
                node_color = self._raw_data[:, idx]
        elif len(node_color) == len(self.point_mask):
            node_color = node_color[self.point_mask]

        # Extract edge color data
        if isinstance(edge_color, str) and edge_color in self._edges.dtype.names:
            edge_color = self._edges[edge_color]

        # Compute or extract layout
        self._xs = np.nan * np.ones(len(self.point_mask))
        self._ys = np.nan * np.ones(len(self.point_mask))
        if positions is None:
            try:
                import networkx as nx
            except ImportError:
                raise ImportError(
                    "You must install the networkx to compute a sfdp layout."
                )
            if self._pos is None:
                g = nx.Graph()
                for row in self._edges:
                    g.add_edge(
                        row["parent"],
                        row["child"],
                        weight=1 / row["mutual_reachability"],
                    )
                self._pos = nx.nx_agraph.graphviz_layout(g, prog="sfdp")
            for k, v in self._pos.items():
                self._xs[k] = v[0]
                self._ys[k] = v[1]
        else:
            if positions.shape[0] == len(self.point_mask):
                self._xs = positions[:, 0]
                self._ys = positions[:, 1]
            elif positions.shape[0] == len(self._points):
                for i, d in enumerate(self._points["id"]):
                    self._xs[d, 0] = positions[i, 0]
                    self._ys[d, 1] = positions[i, 1]
            else:
                raise ValueError("Incorrect number of positions specified.")
        source = self._edges["parent"]
        target = self._edges["child"]
        lc = mc.LineCollection(
            list(
                zip(
                    zip(self._xs[source], self._ys[source]),
                    zip(self._xs[target], self._ys[target]),
                )
            ),
            alpha=edge_alpha,
            cmap=edge_cmap,
            linewidths=edge_width,
            zorder=0,
        )
        lc.set_clim(edge_vmin, edge_vmax)
        if isinstance(edge_color, str):
            lc.set_edgecolor(edge_color)
        else:
            lc.set_array(edge_color)
        if edge_alpha is not None:
            lc.set_alpha(edge_alpha)
        plt.gca().add_collection(lc)
        plt.scatter(
            self._xs[~self.point_mask],
            self._ys[~self.point_mask],
            node_size,
            color="silver",
            marker=node_marker,
            alpha=node_alpha,
            linewidth=0,
            edgecolor="none",
        )
        plt.scatter(
            self._xs[self.point_mask],
            self._ys[self.point_mask],
            node_size,
            node_color,
            cmap=node_cmap,
            marker=node_marker,
            alpha=node_alpha,
            linewidth=0,
            edgecolor="none",
            vmin=node_vmin,
            vmax=node_vmax,
        )
        plt.axis("off")

    def to_numpy(self):
        """Converts the approximation graph to numpy arrays.

        Returns
        -------
        points : np.recarray, shape (n_points, 8)
            A numpy record array with for each point its:
            - id (row index),
            - label,
            - probability,
            - cluster label,
            - cluster probability,
            - cluster centrality,
            - branch label,
            - branch probability

        edges : np.recarray, shape (n_edges, 5)
            A numpy record array with for each edge its:
            - parent point,
            - child point,
            - cluster centrality,
            - mutual reachability,
            - cluster label
        """
        return self._points, self._edges

    def to_pandas(self):
        """Converts the approximation graph to pandas data frames.

        Returns
        -------
        points : pd.DataFrame, shape (n_points, 8)
            A DataFrame with for each point its:
            - id (row index),
            - label,
            - probability,
            - cluster label,
            - cluster probability,
            - cluster centrality,
            - branch label,
            - branch probability

        edges : pd.DataFrame, shape (n_edges, 5)
            A DataFrame with for each edge its:
            - parent point,
            - child point,
            - cluster centrality,
            - mutual reachability,
            - cluster label
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        points = DataFrame(self._points)
        edges = DataFrame(self._edges)
        return points, edges

    def to_networkx(self, feature_names=None):
        """Convert to a NetworkX Graph object.

        Parameters
        ----------
        feature_names : list[n_features]
            Names to use for the data features if available.

        Returns
        -------
        g : nx.Graph
            A NetworkX Graph object containing the non-noise points and edges
            within clusters.

            Node attributes:
            - label,
            - probability,
            - cluster label,
            - cluster probability,
            - cluster centrality,
            - branch label,
            - branch probability,

            Edge attributes:
            - weight (1 / mutual_reachability),
            - mutual_reachability,
            - centrality,
            - cluster label,
            -
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        g = nx.Graph()
        # Add edges
        for row in self._edges:
            g.add_edge(
                row["parent"],
                row["child"],
                weight=1 / row["mutual_reachability"],
                mutual_reachability=row["mutual_reachability"],
                centrality=row["centrality"],
                cluster=row["cluster"],
            )

        # Add FLASC features
        for attr in self._points.dtype.names[1:]:
            nx.set_node_attributes(g, dict(self._points[["id", attr]]), attr)

        # Add raw data features
        if self._raw_data is not None:
            if feature_names is None:
                feature_names = [f"feature {i}" for i in range(self._raw_data.shape[1])]
            for idx, name in enumerate(feature_names):
                nx.set_node_attributes(
                    g,
                    dict(zip(self._points["id"], self._raw_data[:, idx])),
                    name,
                )

        return g


class _BaseCondensedTree:
    """Class with conversion functions shared between both types of condensed
    tree.

    Parameters
    ----------
    raw_tree : numpy recarray from :class:`~hdbscan.HDBSCAN`
        The raw numpy rec array version of the condensed tree as produced
        internally by hdbscan.
    """

    def __init__(self, raw_tree):
        self._raw_tree = raw_tree

    def to_numpy(self):
        """Return a numpy structured array representation of the condensed tree."""
        return self._raw_tree.copy()

    def to_pandas(self):
        """Return a pandas dataframe representation of the condensed tree.

        Each row of the dataframe corresponds to an edge in the tree.
        The columns of the dataframe are `parent`, `child`, `lambda_val`
        and `child_size`.

        The `parent` and `child` are the ids of the
        parent and child nodes in the tree. Node ids less than the number
        of points in the original dataset represent individual points, while
        ids greater than the number of points are clusters.

        The `lambda_val` value is the value (1/distance) at which the `child`
        node leaves the cluster.

        The `child_size` is the number of points in the `child` node.
        """
        try:
            from pandas import DataFrame, Series
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        result = DataFrame(self._raw_tree)
        return result

    def to_networkx(self):
        """Return a NetworkX DiGraph object representing the condensed tree.

        Edge weights in the graph are the lamba values at which child nodes
        'leave' the parent cluster.

        Nodes have a `size` attribute attached giving the number of points
        that are in the cluster (or 1 if it is a singleton point) at the
        point of cluster creation (fewer points may be in the cluster at
        larger lambda values).
        """
        try:
            from networkx import DiGraph, set_node_attributes
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        result = DiGraph()
        for row in self._raw_tree:
            result.add_edge(row["parent"], row["child"], weight=row["lambda_val"])

        set_node_attributes(
            result, dict(self._raw_tree[["child", "child_size"]]), "size"
        )

        return result


class BranchCondensedTree(_BaseCondensedTree):
    """The branch condensed tree structure, which provides a simplified or
    smoothed version of the :class:`~hdbscan.plots.SingleLinkageTree`.

    Parameters
    ----------
    condensed_tree_array : numpy recarray from :class:`~hdbscan.HDBSCAN`
        The raw numpy rec array version of the condensed tree as produced
        internally by hdbscan.

    labels : numpy array
        The final labels for each point in the dataset.

    cluster_points : numpy array
        An array listing point indices of the cluster this tree is for. Used to
        map from within-cluster indices to dataset indices.

    branch_selection_method : string, optional (default 'eom')
        The method of selecting clusters. One of 'eom' or 'leaf'

    allow_single_branch : Boolean, optional (default False)
        Whether to allow the root branch as the only selected branch
    """

    def __init__(
        self,
        condensed_tree_array,
        cluster_points,
        labels,
        branch_labels,
        cluster_labels,
        branch_selection_method="eom",
        allow_single_branch=False,
    ):
        super().__init__(condensed_tree_array)
        self.num_points = labels.shape[0]
        self.labels = labels
        self.branch_labels = branch_labels
        self.cluster_labels = cluster_labels
        self.cluster_points = cluster_points
        self.branch_selection_method = branch_selection_method
        self.allow_single_branch = allow_single_branch

    def get_plot_data(
        self,
        log_size=False,
        leaf_separation=1,
        max_rectangle_per_icicle=20,
        label_for="final",
    ):
        """
        Generates data for use in plotting the 'icicle plot' or dendrogram
        plot of a branch condensed tree generated by HDBSCAN.

        Parameters
        ----------
        labels : numpy array
            The subgroup-label of each point.

        log_size : boolean, optional
            Use log scale for the 'size' of clusters (i.e. number of
            points in the cluster at a given lambda value). (default
            False)

        leaf_separation : float, optional
            How far apart to space the final leaves of the
            dendrogram. (default 1)

        max_rectangles_per_icicle : int, optional
            To simplify the plot this method will only emit
            ``max_rectangles_per_icicle`` bars per branch of the dendrogram.
            This ensures that we don't suffer from massive overplotting in
            cases with a lot of data points.

        label_for : str
            Specify which labels to use: 'final', 'branch', 'cluster'.

        Returns
        -------
        plot_data : dict
            Data associated to bars in a bar plot:
                `bar_centers` x coordinate centers for bars
                `bar_tops` heights of bars in lambda scale
                `bar_bottoms` y coordinate of bottoms of bars
                `bar_widths` widths of the bars (in x coord scale)
                `bar_labels` data point label for each bar
                `cluster_bounds` a 4-tuple of [left, right, bottom, top] giving
                the bounds on a full set of cluster bars
            Data associates with cluster splits:
                `line_xs` x coordinates for horizontal dendrogram lines
                `line_ys` y coordinates for horizontal dendrogram lines
        label_start : int
            The minimum label value (excluding noise labels).
        label_end : int
            The maximum label value (excluding noise labels).
        """
        leaves = _get_leaves(self._raw_tree)
        cluster_tree = self._raw_tree[self._raw_tree["child_size"] > 1]
        last_leaf = self._raw_tree["parent"].max()
        root = self._raw_tree["parent"].min()
        if label_for == "final":
            labels = self.labels[self.cluster_points]
        elif label_for == "branch":
            labels = self.branch_labels[self.cluster_points]
        elif label_for == "cluster":
            labels = self.cluster_labels[self.cluster_points]
        else:
            raise ValueError("label_for must be one of 'final', 'branch', 'cluster'")

        # We want to get the x and y coordinates for the start of each cluster
        # Initialize the leaves, since we know where they go, the iterate
        # through everything from the leaves back, setting coords as we go
        if isinstance(leaves, np.int64):
            cluster_x_coords = {leaves: leaf_separation}
        else:
            cluster_x_coords = dict(
                zip(leaves, [leaf_separation * x for x in range(len(leaves))])
            )
        cluster_y_coords = {root: 0.0}

        # print(last_leaf, root)
        for cluster in range(last_leaf, root - 1, -1):
            split = self._raw_tree[["child", "lambda_val"]]
            split = split[
                (self._raw_tree["parent"] == cluster)
                & (self._raw_tree["child_size"] > 1)
            ]
            # print(cluster, len(split))
            if len(split["child"]) > 1:
                left_child, right_child = split["child"]
                cluster_x_coords[cluster] = np.mean(
                    [
                        cluster_x_coords[left_child],
                        cluster_x_coords[right_child],
                    ]
                )
                cluster_y_coords[left_child] = split["lambda_val"][0]
                cluster_y_coords[right_child] = split["lambda_val"][1]

        # We use bars to plot the 'icicles', so we need to generate centers, tops,
        # bottoms and widths for each rectangle. We can go through each cluster
        # and do this for each in turn.
        bar_centers = []
        bar_tops = []
        bar_bottoms = []
        bar_widths = []
        bar_labels = []

        cluster_bounds = {}

        scaling = np.sum(self._raw_tree[self._raw_tree["parent"] == root]["child_size"])

        if log_size:
            scaling = np.log(scaling)

        for c in range(last_leaf, root - 1, -1):
            cluster_bounds[c] = [0, 0, 0, 0]

            c_children = self._raw_tree[self._raw_tree["parent"] == c]
            point_children = c_children["child"][c_children["child"] < root]
            if len(point_children) == 0:
                label = 0
            else:
                label = labels[point_children[0]]
            current_size = np.sum(c_children["child_size"])
            current_lambda = cluster_y_coords[c]
            cluster_max_size = current_size
            cluster_max_lambda = c_children["lambda_val"].max()
            cluster_min_size = np.sum(
                c_children[c_children["lambda_val"] == cluster_max_lambda]["child_size"]
            )

            if log_size:
                current_size = np.log(current_size)
                cluster_max_size = np.log(cluster_max_size)
                cluster_min_size = np.log(cluster_min_size)

            total_size_change = float(cluster_max_size - cluster_min_size)
            step_size_change = total_size_change / max_rectangle_per_icicle

            cluster_bounds[c][CB_LEFT] = cluster_x_coords[c] * scaling - (
                current_size / 2.0
            )
            cluster_bounds[c][CB_RIGHT] = cluster_x_coords[c] * scaling + (
                current_size / 2.0
            )
            cluster_bounds[c][CB_BOTTOM] = cluster_y_coords[c]
            cluster_bounds[c][CB_TOP] = np.max(c_children["lambda_val"])

            last_step_size = current_size
            last_step_lambda = current_lambda

            for i in np.argsort(c_children["lambda_val"]):
                row = c_children[i]
                if row["lambda_val"] != current_lambda and (
                    last_step_size - current_size > step_size_change
                    or row["lambda_val"] == cluster_max_lambda
                ):
                    bar_centers.append(cluster_x_coords[c] * scaling)
                    bar_tops.append(row["lambda_val"] - last_step_lambda)
                    bar_bottoms.append(last_step_lambda)
                    bar_widths.append(last_step_size)
                    bar_labels.append(label)
                    last_step_size = current_size
                    last_step_lambda = current_lambda
                if log_size:
                    exp_size = np.exp(current_size) - row["child_size"]
                    # Ensure we don't try to take log of zero
                    if exp_size > 0.01:
                        current_size = np.log(np.exp(current_size) - row["child_size"])
                    else:
                        current_size = 0.0
                else:
                    current_size -= row["child_size"]
                current_lambda = row["lambda_val"]

        # Finally we need the horizontal lines that occur at cluster splits.
        line_xs = []
        line_ys = []

        for row in self._raw_tree[self._raw_tree["child_size"] > 1]:
            parent = row["parent"]
            child = row["child"]
            child_size = row["child_size"]
            if log_size:
                child_size = np.log(child_size)
            sign = np.sign(cluster_x_coords[child] - cluster_x_coords[parent])
            line_xs.append(
                [
                    cluster_x_coords[parent] * scaling,
                    cluster_x_coords[child] * scaling + sign * (child_size / 2.0),
                ]
            )
            line_ys.append([cluster_y_coords[child], cluster_y_coords[child]])

        return (
            {
                "bar_centers": bar_centers,
                "bar_tops": bar_tops,
                "bar_bottoms": bar_bottoms,
                "bar_widths": bar_widths,
                "bar_labels": bar_labels,
                "line_xs": line_xs,
                "line_ys": line_ys,
                "cluster_bounds": cluster_bounds,
            },
            labels[labels >= 0].min(),
            labels.max(),
        )

    def _select_clusters(self):
        """Recovers selected branches respecting selection parameters."""
        selected = None
        cluster_tree = self._raw_tree[self._raw_tree["child_size"] > 1]
        if self.branch_selection_method == "eom":
            stability = compute_stability(self._raw_tree)
            if self.allow_single_branch:
                node_list = sorted(stability.keys(), reverse=True)
            else:
                node_list = sorted(stability.keys(), reverse=True)[:-1]
            is_cluster = {cluster: True for cluster in node_list}

            for node in node_list:
                child_selection = cluster_tree["parent"] == node
                subtree_stability = np.sum(
                    [
                        stability[child]
                        for child in cluster_tree["child"][child_selection]
                    ]
                )

                if subtree_stability > stability[node]:
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    for sub_node in _bfs_from_cluster_tree(cluster_tree, node):
                        if sub_node != node:
                            is_cluster[sub_node] = False

            selected = [cluster for cluster in is_cluster if is_cluster[cluster]]
        else:
            selected = _get_leaves(self._raw_tree)

        return sorted(selected)

    def plot(
        self,
        leaf_separation=1,
        label_clusters=False,
        selection_palette=None,
        axis=None,
        log_size=False,
        max_rectangles_per_icicle=20,
        label_offset_factor=0.7,
        label_for="final",
        color_centre_as_noise=False,
    ):
        """Use matplotlib to plot an 'icicle plot' dendrogram of the condensed tree.

        Effectively this is a dendrogram where the width of each cluster bar is
        equal to the number of points (or log of the number of points) in the cluster
        at the given lambda value. Thus bars narrow as points progressively drop
        out of clusters. Bars are coloured and labelled by the selected branches.

        Parameters
        ----------
        leaf_separation : float, optional (default 1)
            How far apart to space the final leaves of the dendrogram.

        label_clusters : boolean, optional (default True)
            If select_clusters is True then this determines whether to draw text
            labels on the clusters.

        selection_palette : list of colors, optional (default None)
            If not None, and at least as long as the number of clusters, draw
            ovals in colors iterating through this palette. This can aid in
            cluster identification when plotting.

        axis : matplotlib axis or None, optional (default None)
            The matplotlib axis to render to. If None then a new axis will be
            generated. The rendered axis will be returned.

        log_size : boolean, optional (default False)
            Use log scale for the 'size' of clusters (i.e. number of points in
            the cluster at a given lambda value).

        max_rectangles_per_icicle : int, optional (default 20)
            To simplify the plot this method will only emit
            ``max_rectangles_per_icicle`` bars per branch of the dendrogram.
            This ensures that we don't suffer from massive overplotting in cases
            with a lot of data points.

        label_offset_factor : float, optional (default 0.7)
            Controls height-wise offset of cluster labels.

        label_for : str
            Specify which labels to use: 'final', 'branch', 'cluster'.

        color_centre_as_noise : boolean, optional (default False)
            If True, central points are coloured as noise points in the
            cluster condensed tree.

        Returns
        -------
        axis : matplotlib axis
            The axis on which the 'icicle plot' has been rendered.
        """
        try:
            import matplotlib as mc
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "You must install the matplotlib library to plot the condensed tree."
                "Use get_plot_data to calculate the relevant data without plotting."
            )

        plot_data, label_start, label_end = self.get_plot_data(
            log_size=log_size,
            leaf_separation=leaf_separation,
            max_rectangle_per_icicle=max_rectangles_per_icicle,
            label_for=label_for,
        )

        if axis is None:
            axis = plt.gca()
        if selection_palette is None:
            selection_palette = mc.colormaps["tab10"].colors

        if color_centre_as_noise:
            bar_colors = [
                (
                    selection_palette[l % len(selection_palette)]
                    if l >= 0 and l < label_end
                    else "silver"
                )
                for l in plot_data["bar_labels"]
            ]
        else:
            bar_colors = [
                (selection_palette[l % len(selection_palette)] if l >= 0 else "silver")
                for l in plot_data["bar_labels"]
            ]

        axis.bar(
            plot_data["bar_centers"],
            plot_data["bar_tops"],
            bottom=plot_data["bar_bottoms"],
            width=plot_data["bar_widths"],
            color=bar_colors,
            align="center",
            linewidth=0,
        )

        drawlines = []
        for xs, ys in zip(plot_data["line_xs"], plot_data["line_ys"]):
            drawlines.append(xs)
            drawlines.append(ys)
        axis.plot(*drawlines, color="black", linewidth=1)

        if label_clusters:
            chosen_clusters = self._select_clusters()

            # Extract the chosen cluster bounds. If enough duplicate data points exist in the
            # data the lambda value might be infinite. This breaks labeling and highlighting
            # the chosen clusters.
            cluster_bounds = np.array(
                [plot_data["cluster_bounds"][c] for c in chosen_clusters]
            )
            if not np.isfinite(cluster_bounds).all():
                warn(
                    "Infinite lambda values encountered in chosen clusters."
                    " This might be due to duplicates in the data."
                )

            # Extract the plot range of the y-axis and set default center and height values for ellipses.
            # Extremly dense clusters might result in near infinite lambda values. Setting max_height
            # based on the percentile should alleviate the impact on plotting.
            plot_range = np.hstack([plot_data["bar_tops"], plot_data["bar_bottoms"]])
            plot_range = plot_range[np.isfinite(plot_range)]
            mean_y_center = np.mean([np.max(plot_range), np.min(plot_range)])
            max_height = np.diff(np.percentile(plot_range, q=[10, 90]))

            for i, c in enumerate(chosen_clusters):
                c_bounds = plot_data["cluster_bounds"][c]
                height = c_bounds[CB_TOP] - c_bounds[CB_BOTTOM]
                center = (
                    np.mean([c_bounds[CB_LEFT], c_bounds[CB_RIGHT]]),
                    np.mean([c_bounds[CB_TOP], c_bounds[CB_BOTTOM]]),
                )

                # Set center and height to default values if necessary
                if not np.isfinite(center[1]):
                    center = (center[0], mean_y_center)
                if not np.isfinite(height):
                    height = max_height

                axis.annotate(
                    str(i + label_start),
                    xy=center,
                    xytext=(
                        center[0],
                        center[1] + label_offset_factor * height,
                    ),
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        axis.set_xticks([])
        for side in ("right", "top", "bottom"):
            axis.spines[side].set_visible(False)
        axis.invert_yaxis()
        axis.set_ylabel("$e$ value")

        return axis


class ClusterCondensedTree(_BaseCondensedTree):
    """The cluster condensed tree structure, which provides a simplified or
    smoothed version of the :class:`~hdbscan.plots.SingleLinkageTree`.

    Parameters
    ----------
    condensed_tree_array : numpy recarray from :class:`~hdbscan.HDBSCAN`
        The raw numpy rec array version of the condensed tree as produced
        internally by hdbscan.

    cluster_labels : numpy array
        The cluster labels for each point in the dataset.

    cluster_selection_method : string, optional (default 'eom')
        The method of selecting clusters. One of 'eom' or 'leaf'

    cluster_selection_epsilon : float, optional (default 0)
        The applied epsilon value for selecting clusters.

    allow_single_cluster : Boolean, optional (default False)
        Whether to allow the root cluster as the only selected cluster
    """

    def __init__(
        self,
        condensed_tree_array,
        cluster_labels,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0,
        allow_single_cluster=False,
    ):
        super().__init__(condensed_tree_array)
        self.labels = cluster_labels
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.allow_single_cluster = allow_single_cluster

    def get_plot_data(
        self, leaf_separation=1, log_size=False, max_rectangle_per_icicle=20
    ):
        """Generates data for use in plotting the 'icicle plot' or dendrogram
        plot of the condensed tree generated by HDBSCAN.

        Parameters
        ----------
        leaf_separation : float, optional
            How far apart to space the final leaves of the dendrogram. (default
            1)

        log_size : boolean, optional
            Use log scale for the 'size' of clusters (i.e. number of points in
            the cluster at a given lambda value). (default False)

        max_rectangles_per_icicle : int, optional
            To simplify the plot this method will only emit
            ``max_rectangles_per_icicle`` bars per branch of the dendrogram.
            This ensures that we don't suffer from massive overplotting in cases
            with a lot of data points.

        Returns
        -------
        plot_data : dict
            Data associated to bars in a bar plot:
                `bar_centers` x coordinate centers for bars `bar_tops` heights
                of bars in lambda scale `bar_bottoms` y coordinate of bottoms of
                bars `bar_widths` widths of the bars (in x coord scale)
                `bar_labels` cluster label for each bar `cluster_bounds` a
                4-tuple of [left, right, bottom, top] giving
                    the bounds on a full set of cluster bars
            Data associates with cluster splits:
                `line_xs` x coordinates for horizontal dendrogram lines
                `line_ys` y coordinates for horizontal dendrogram lines
        """
        leaves = _get_leaves(self._raw_tree)
        last_leaf = self._raw_tree["parent"].max()
        root = self._raw_tree["parent"].min()

        # We want to get the x and y coordinates for the start of each cluster
        # Initialize the leaves, since we know where they go, the iterate
        # through everything from the leaves back, setting coords as we go
        if isinstance(leaves, np.int64):
            cluster_x_coords = {leaves: leaf_separation}
        else:
            cluster_x_coords = dict(
                zip(leaves, [leaf_separation * x for x in range(len(leaves))])
            )
        cluster_y_coords = {root: 0.0}

        for cluster in range(last_leaf, root - 1, -1):
            split = self._raw_tree[["child", "lambda_val"]]
            split = split[
                (self._raw_tree["parent"] == cluster)
                & (self._raw_tree["child_size"] > 1)
            ]
            if len(split["child"]) > 1:
                left_child, right_child = split["child"]
                cluster_x_coords[cluster] = np.mean(
                    [cluster_x_coords[left_child], cluster_x_coords[right_child]]
                )
                cluster_y_coords[left_child] = split["lambda_val"][0]
                cluster_y_coords[right_child] = split["lambda_val"][1]

        # We use bars to plot the 'icicles', so we need to generate centers,
        # tops, bottoms and widths for each rectangle. We can go through each
        # cluster and do this for each in turn.
        bar_centers = []
        bar_tops = []
        bar_bottoms = []
        bar_widths = []
        bar_labels = []

        cluster_bounds = {}

        scaling = np.sum(self._raw_tree[self._raw_tree["parent"] == root]["child_size"])

        if log_size:
            scaling = np.log(scaling)

        for c in range(last_leaf, root - 1, -1):
            cluster_bounds[c] = [0, 0, 0, 0]

            c_children = self._raw_tree[self._raw_tree["parent"] == c]
            point_children = c_children["child"][c_children["child"] < root]
            if len(point_children) == 0:
                label = 0
            else:
                label = self.labels[point_children[0]]
            current_size = np.sum(c_children["child_size"])
            current_lambda = cluster_y_coords[c]
            cluster_max_size = current_size
            cluster_max_lambda = c_children["lambda_val"].max()
            cluster_min_size = np.sum(
                c_children[c_children["lambda_val"] == cluster_max_lambda]["child_size"]
            )

            if log_size:
                current_size = np.log(current_size)
                cluster_max_size = np.log(cluster_max_size)
                cluster_min_size = np.log(cluster_min_size)

            total_size_change = float(cluster_max_size - cluster_min_size)
            step_size_change = total_size_change / max_rectangle_per_icicle

            cluster_bounds[c][CB_LEFT] = cluster_x_coords[c] * scaling - (
                current_size / 2.0
            )
            cluster_bounds[c][CB_RIGHT] = cluster_x_coords[c] * scaling + (
                current_size / 2.0
            )
            cluster_bounds[c][CB_BOTTOM] = cluster_y_coords[c]
            cluster_bounds[c][CB_TOP] = np.max(c_children["lambda_val"])

            last_step_size = current_size
            last_step_lambda = current_lambda

            for i in np.argsort(c_children["lambda_val"]):
                row = c_children[i]
                if row["lambda_val"] != current_lambda and (
                    last_step_size - current_size > step_size_change
                    or row["lambda_val"] == cluster_max_lambda
                ):
                    bar_centers.append(cluster_x_coords[c] * scaling)
                    bar_tops.append(row["lambda_val"] - last_step_lambda)
                    bar_bottoms.append(last_step_lambda)
                    bar_widths.append(last_step_size)
                    bar_labels.append(label)
                    last_step_size = current_size
                    last_step_lambda = current_lambda
                if log_size:
                    exp_size = np.exp(current_size) - row["child_size"]
                    # Ensure we don't try to take log of zero
                    if exp_size > 0.01:
                        current_size = np.log(np.exp(current_size) - row["child_size"])
                    else:
                        current_size = 0.0
                else:
                    current_size -= row["child_size"]
                current_lambda = row["lambda_val"]

        # Finally we need the horizontal lines that occur at cluster splits.
        line_xs = []
        line_ys = []

        for row in self._raw_tree[self._raw_tree["child_size"] > 1]:
            parent = row["parent"]
            child = row["child"]
            child_size = row["child_size"]
            if log_size:
                child_size = np.log(child_size)
            sign = np.sign(cluster_x_coords[child] - cluster_x_coords[parent])
            line_xs.append(
                [
                    cluster_x_coords[parent] * scaling,
                    cluster_x_coords[child] * scaling + sign * (child_size / 2.0),
                ]
            )
            line_ys.append([cluster_y_coords[child], cluster_y_coords[child]])

        return {
            "bar_centers": bar_centers,
            "bar_tops": bar_tops,
            "bar_bottoms": bar_bottoms,
            "bar_widths": bar_widths,
            "bar_labels": bar_labels,
            "line_xs": line_xs,
            "line_ys": line_ys,
            "cluster_bounds": cluster_bounds,
        }

    def _select_clusters(self):
        """Recovers selected clusters respecting selection parameters."""
        selected = None
        cluster_tree = self._raw_tree[self._raw_tree["child_size"] > 1]
        if self.cluster_selection_method == "eom":
            stability = compute_stability(self._raw_tree)
            if self.allow_single_cluster:
                node_list = sorted(stability.keys(), reverse=True)
            else:
                node_list = sorted(stability.keys(), reverse=True)[:-1]
            is_cluster = {cluster: True for cluster in node_list}

            for node in node_list:
                child_selection = cluster_tree["parent"] == node
                subtree_stability = np.sum(
                    [
                        stability[child]
                        for child in cluster_tree["child"][child_selection]
                    ]
                )

                if subtree_stability > stability[node]:
                    is_cluster[node] = False
                    stability[node] = subtree_stability
                else:
                    for sub_node in _bfs_from_cluster_tree(cluster_tree, node):
                        if sub_node != node:
                            is_cluster[sub_node] = False

            selected = [cluster for cluster in is_cluster if is_cluster[cluster]]
        else:
            selected = _get_leaves(self._raw_tree)

        if self.cluster_selection_epsilon != 0.0:
            selected = epsilon_search(
                set(selected),
                cluster_tree.copy(),
                self.cluster_selection_epsilon,
                self.allow_single_cluster,
            )
        return sorted(selected)

    def plot(
        self,
        leaf_separation=1,
        label_clusters=False,
        selection_palette=None,
        axis=None,
        log_size=False,
        max_rectangles_per_icicle=20,
        label_offset_factor=0.7,
    ):
        """Use matplotlib to plot an 'icicle plot' dendrogram of the condensed tree.

        Effectively this is a dendrogram where the width of each cluster bar is
        equal to the number of points (or log of the number of points) in the cluster
        at the given lambda value. Thus bars narrow as points progressively drop
        out of clusters. Bars are coloured and labelled by the selected branches.

        Parameters
        ----------
        leaf_separation : float, optional (default 1)
            How far apart to space the final leaves of the dendrogram.

        label_clusters : boolean, optional (default True)
            If select_clusters is True then this determines whether to draw text
            labels on the clusters.

        selection_palette : list of colors, optional (default None)
            If not None, and at least as long as the number of clusters, draw
            ovals in colors iterating through this palette. This can aid in
            cluster identification when plotting.

        axis : matplotlib axis or None, optional (default None)
            The matplotlib axis to render to. If None then a new axis will be
            generated. The rendered axis will be returned.

        log_size : boolean, optional (default False)
            Use log scale for the 'size' of clusters (i.e. number of points in
            the cluster at a given lambda value).

        max_rectangles_per_icicle : int, optional (default 20)
            To simplify the plot this method will only emit
            ``max_rectangles_per_icicle`` bars per branch of the dendrogram.
            This ensures that we don't suffer from massive overplotting in cases
            with a lot of data points.

        label_offset_factor : float, optional (default 0.7)
            Controls height-wise offset of cluster labels.

        Returns
        -------
        axis : matplotlib axis
            The axis on which the 'icicle plot' has been rendered.
        """
        try:
            import matplotlib as mc
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "You must install the matplotlib library to plot the condensed tree."
                "Use get_plot_data to calculate the relevant data without plotting."
            )

        plot_data = self.get_plot_data(
            log_size=log_size,
            leaf_separation=leaf_separation,
            max_rectangle_per_icicle=max_rectangles_per_icicle,
        )

        if axis is None:
            axis = plt.gca()
        if selection_palette is None:
            selection_palette = mc.colormaps["tab10"].colors

        axis.bar(
            plot_data["bar_centers"],
            plot_data["bar_tops"],
            bottom=plot_data["bar_bottoms"],
            width=plot_data["bar_widths"],
            color=[
                (selection_palette[l % len(selection_palette)] if l >= 0 else "silver")
                for l in plot_data["bar_labels"]
            ],
            align="center",
            linewidth=0,
        )

        drawlines = []
        for xs, ys in zip(plot_data["line_xs"], plot_data["line_ys"]):
            drawlines.append(xs)
            drawlines.append(ys)
        axis.plot(*drawlines, color="black", linewidth=1)

        if label_clusters:
            chosen_clusters = self._select_clusters()

            # Extract the chosen cluster bounds. If enough duplicate data points exist in the
            # data the lambda value might be infinite. This breaks labeling and highlighting
            # the chosen clusters.
            cluster_bounds = np.array(
                [plot_data["cluster_bounds"][c] for c in chosen_clusters]
            )
            if not np.isfinite(cluster_bounds).all():
                warn(
                    "Infinite lambda values encountered in chosen clusters."
                    " This might be due to duplicates in the data."
                )

            # Extract the plot range of the y-axis and set default center and height values for ellipses.
            # Extremly dense clusters might result in near infinite lambda values. Setting max_height
            # based on the percentile should alleviate the impact on plotting.
            plot_range = np.hstack([plot_data["bar_tops"], plot_data["bar_bottoms"]])
            plot_range = plot_range[np.isfinite(plot_range)]
            mean_y_center = np.mean([np.max(plot_range), np.min(plot_range)])
            max_height = np.diff(np.percentile(plot_range, q=[10, 90]))

            for i, c in enumerate(chosen_clusters):
                c_bounds = plot_data["cluster_bounds"][c]
                height = c_bounds[CB_TOP] - c_bounds[CB_BOTTOM]
                center = (
                    np.mean([c_bounds[CB_LEFT], c_bounds[CB_RIGHT]]),
                    np.mean([c_bounds[CB_TOP], c_bounds[CB_BOTTOM]]),
                )

                # Set center and height to default values if necessary
                if not np.isfinite(center[1]):
                    center = (center[0], mean_y_center)
                if not np.isfinite(height):
                    height = max_height

                axis.annotate(
                    str(i),
                    xy=center,
                    xytext=(
                        center[0],
                        center[1] + label_offset_factor * height,
                    ),
                    horizontalalignment="center",
                    verticalalignment="top",
                )

        axis.set_xticks([])
        for side in ("right", "top", "bottom"):
            axis.spines[side].set_visible(False)
        axis.invert_yaxis()
        axis.set_ylabel("$\lambda$ value")

        return axis
