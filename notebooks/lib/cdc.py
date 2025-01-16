"""
This module implements the Clustering by Direction Centrality (CDD) clustering
algorithm from [1]_.

- Original author: Dehua Peng
- Original code at: https://github.com/ZPGuiGroupWhu/ClusteringDirectionCentrality/blob/master/Toolkit/Python_2D/CDC.py 
- Adapted for evaluation in pyflasc by: Jelmer Bot
- License: Apache 2.0 License.

Reference
---------
.. [1] Peng, D., Gui, Z., Wang, D. et al. Clustering by measuring local
   direction centrality for data with heterogeneous density and weak
   connectivity. Nat.Commun. 13, 5455 (2022).
   https://www.nature.com/articles/s41467-022-33136-9
"""

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, ClusterMixin


def cdc(X, k, ratio):
    num = len(X)
    indices = (
        NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
        .fit(X)
        .kneighbors(X, return_distance=False)[:, 1 : k + 1]
    )

    # Visit neighbors and store their angle
    angle = np.empty((num, k), dtype=np.float32)
    for i in range(num):
        for j in range(k):
            delta_x = X[indices[i, j], 0] - X[i, 0]
            delta_y = X[indices[i, j], 1] - X[i, 1]
            if delta_x == 0:
                if delta_y == 0:
                    angle[i, j] = 0
                elif delta_y > 0:
                    angle[i, j] = math.pi / 2
                else:
                    angle[i, j] = 3 * math.pi / 2
            elif delta_x > 0:
                if math.atan(delta_y / delta_x) >= 0:
                    angle[i, j] = math.atan(delta_y / delta_x)
                else:
                    angle[i, j] = 2 * math.pi + math.atan(delta_y / delta_x)
            else:
                angle[i, j] = math.pi + math.atan(delta_y / delta_x)

    # Aggregate angles per point
    angle.sort()
    angle_var = np.zeros(num, dtype=np.float32)
    angle_var = np.sum((np.diff(angle, axis=-1) - 2 * math.pi / k) ** 2, axis=-1)
    angle_var += ((angle[:, 0] - angle[:, -1] + 2 * math.pi) - 2 * math.pi / k) ** 2
    angle_var /= k
    angle_var /= (k - 1) * 4 * pow(math.pi, 2) / pow(k, 2)

    # Mark boundary points using angle threshold
    threshold_idx = math.ceil(num * ratio)
    threshold_value = np.partition(angle_var, threshold_idx)[threshold_idx]
    is_internal = angle_var < threshold_value

    # Compute distance to closest boundary point and list closest non boundary point for boundary points
    distance_to_boundary = np.empty(num, dtype=np.float32)
    for i in range(num):
        knn_ind = is_internal[indices[i, :]]
        if is_internal[i]:
            if False in knn_ind:
                bdpts_ind = np.where(~knn_ind)[0]
                bd_id = indices[i, bdpts_ind[0]]
                distance_to_boundary[i] = math.sqrt(
                    sum(pow((X[i, :] - X[bd_id, :]), 2))
                )
            else:
                distance_to_boundary[i] = float("inf")
                for j in range(num):
                    if is_internal[j] == 0:
                        temp_dis = math.sqrt(sum(pow((X[i, :] - X[j, :]), 2)))
                        if temp_dis < distance_to_boundary[i]:
                            distance_to_boundary[i] = temp_dis
        else:
            if True in knn_ind:
                bdpts_ind = np.where(knn_ind)[0]
                bd_id = indices[i, bdpts_ind[0]]
                distance_to_boundary[i] = bd_id
            else:
                mark_dis = float("inf")
                for j in range(num):
                    if is_internal[j] == 1:
                        temp_dis = math.sqrt(sum(pow((X[i, :] - X[j, :]), 2)))
                        if temp_dis < mark_dis:
                            mark_dis = temp_dis
                            distance_to_boundary[i] = j

    # Set cluster labels for internal points
    mark = 1
    cluster = np.zeros(num)
    for i in range(num):
        if is_internal[i] and cluster[i] == 0:
            cluster[i] = mark
            for j in range(num):
                if (
                    is_internal[j]
                    and math.sqrt(sum(pow((X[i, :] - X[j, :]), 2)))
                    <= distance_to_boundary[i] + distance_to_boundary[j]
                ):
                    if cluster[j] == 0:
                        cluster[j] = cluster[i]
                    else:
                        cluster[cluster == cluster[j]] = cluster[i]
            mark = mark + 1

    # Set cluster labels for boundary points
    for i in range(num):
        if not is_internal[i]:
            cluster[i] = cluster[int(distance_to_boundary[i])]

    # Remap cluster labels to be consecutive
    mark = 1
    storage = np.zeros(num)
    for i in range(num):
        if cluster[i] in storage:
            temp_ind = np.where(storage == cluster[i])
            cluster[i] = cluster[temp_ind[0][0]]
        else:
            storage[i] = cluster[i]
            cluster[i] = mark
            mark = mark + 1

    return cluster, is_internal


class CDC(BaseEstimator, ClusterMixin):
    def __init__(self, k=10, ratio=0.1):
        self.k = k
        self.ratio = ratio

    def fit(self, X, y=None):
        self.labels_, self.is_internal_ = cdc(X, self.k, self.ratio)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_
