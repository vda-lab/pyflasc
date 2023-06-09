""" flasc - a Python implementation of FLASC.

FLASC* is a branch-aware clustering algorithm, that builds upon 
:py:mod:`hdbscan` to detect branching structures within clusters. The 
algorithm returns a labelling that separates noise points from clusters and 
branches from each other. In addition, the 
:class:`single-linkage <hdbscan.plots.SingleLinkageTree>` 
and
:class:`condensed linkage <hdbscan.plots.CondensedTree>` 
hierarchies are provided for both the clustering as the branch detection stages.
"""

from ._flasc import flasc
from ._sklearn import FLASC
from . import prediction
from . import plots

__all__ = [
    flasc, FLASC, prediction, plots
]