# Plotting object for the Approximation graph
# Author: Jelmer Bot
# Inspired by the hdbscan.plots module
# License: BSD 3 clause
import numpy as np


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
        self._edges = np.core.records.fromarrays(
            np.hstack(
                (
                    np.concatenate(approximation_graphs),
                    np.repeat(
                        np.arange(len(approximation_graphs)),
                        [g.shape[0] for g in approximation_graphs],
                    )[None].T,
                )
            ).transpose(),
            names="parent, child, centrality, mutual_reachability, cluster",
            formats="intp, intp, double, double, intp",
        )
        self.point_mask = cluster_labels >= 0
        self._raw_data = raw_data[self.point_mask, :] if raw_data is not None else None
        self._points = np.core.records.fromarrays(
            np.vstack(
                (
                    np.where(self.point_mask)[0],
                    labels[self.point_mask],
                    probabilities[self.point_mask],
                    cluster_labels[self.point_mask],
                    cluster_probabilities[self.point_mask],
                    cluster_centralities[self.point_mask],
                    branch_labels[self.point_mask],
                    branch_probabilities[self.point_mask],
                )
            ),
            names="id, label, probability, cluster_label, cluster_probability, cluster_centrality, branch_label, branch_probability",
            formats="intp, intp, double, intp, double, double, intp, double",
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
            neato layout, which requires pygraphviz to be installed and available.

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
            if no ``feature_names`` are given,
            or anything matplotlib scatter interprets as a color.

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
                    if node_vmax is None:
                        node_vmax = 10
                    node_cmap = "tab10"
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
                    "You must install the networkx to compute a neato layout."
                )
            if self._pos is None:
                g = nx.Graph()
                for row in self._edges:
                    g.add_edge(
                        row["parent"],
                        row["child"],
                        weight=1 / row["mutual_reachability"],
                    )
                self._pos = nx.nx_agraph.graphviz_layout(g, prog="neato")
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
            self._xs[self.point_mask],
            self._ys[self.point_mask],
            node_size,
            node_color,
            cmap=node_cmap,
            marker=node_marker,
            alpha=node_alpha,
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
