[![PyPI version](https://badge.fury.io/py/pyflasc.svg)](https://badge.fury.io/py/pyflasc)
[![Tests](https://github.com/vda-lab/pyflasc/actions/workflows/Wheels.yml/badge.svg?branch=main)](https://github.com/vda-lab/pyflasc/actions/workflows/Wheels.yml)
[![DOI](https://zenodo.org/badge/650995702.svg)](https://zenodo.org/doi/10.5281/zenodo.13326222)

# FLASC: Flare-Sensitive Clustering

FLASC - Flare-Sensitive Clustering, adds an efficient post-processing step to
the [HDBSCAN\*](https://github.com/scikit-learn-contrib/hdbscan) density-based
clustering algorithm to detect branching structures within clusters.

The algorithm adds two parameters that may need tuning with respect to
HDBSCAN\*, but both are intuitive to tune: minimum branch size and branch
selection strategy.

## How to use FLASC

The FLASC package is closely based on the HDBSCAN* package and supports the same
API, except sparse inputs, which are not supported yet.

```python
from flasc import FLASC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.load('./notebooks/data/flared/flared_clusterable_data.npy')
clusterer = FLASC(min_cluster_size=15)
clusterer.fit(data)
colors = sns.color_palette('tab10', 10)
point_colors = [
  sns.desaturate(colors[l], p)
  for l, p in zip(clusterer.labels_, clusterer.probabilities_)
]
plt.scatter(data[:, 0], data[:, 1], 2, point_colors, alpha=0.5)
plt.axis('off')
plt.show()
```

![Example point cloud](docs/_static/example.png)

## Example Notebooks

A notebook demonstrating how the algorithm works is available at [How FLASC
Works](https://nbviewer.org/github/vda-lab/pyflasc/blob/master/notebooks/How%20FLASC%20Works.ipynb).
The other notebooks demonstrate the algorithm on several data sets and contain
the analyses presented in our paper.

## Installing

Binary wheels are available on PyPI. Presuming you have an up-to-date pip:

```bash
pip install pyflasc
```
For a manual install of the latest code directly from GitHub:

```bash
pip install --upgrade git+https://github.com/vda-lab/pyflasc.git#egg=pyflasc
```

Alternatively download the package, install requirements, and manually run the
installer:

```bash
wget https://github.com/vda-lab/pyflasc/archive/main.zip
unzip main.zip
rm main.zip
cd flasc-main

pip install -t .
```

## Citing

Please cite [our publication](https://doi.org/10.7717/peerj-cs.2792) when using the algorithm:

    Bot DM, Peeters J, Liesenborgs J, Aerts J. 2025. FLASC: a flare-sensitive
    clustering algorithm. PeerJ Computer Science 11:e2792
    https://doi.org/10.7717/peerj-cs.2792 

in bibtex:

```bibtex
@article{bot2025flasc,
  title   = {{FLASC: a flare-sensitive clustering algorithm}},
  author  = {Bot, Dani{\"{e}}l M. and Peeters, Jannes and Liesenborgs, Jori and Aerts, Jan},
  year    = {2025},
  month   = {apr},
  journal = {PeerJ Comput. Sci.},
  volume  = {11},
  pages   = {e2792},
  issn    = {2376-5992},
  doi     = {10.7717/peerj-cs.2792},
  url     = {https://peerj.com/articles/cs-2792},
}
```

The FLASC algorithm and software package is closely related to McInnes et al.'s
HDBSCAN\* software package. We refer to their [Journal of Open Source Software
article](http://joss.theoj.org/papers/10.21105/joss.00205) and [paper in the
ICDMW 2017 proceedings](https://ieeexplore.ieee.org/abstract/document/8215642/)
for information on how to cite their software package and high-performance
algorithm.

## Licensing

The FLASC package has a 3-Clause BSD license.
