# FLeCS

**F**unctional and **Le**arnable **C**ell dynamic**S**

<img src="docs/flecs_logo.png" alt="flecs_logo" width="300"/>

## Overview

<img src="docs/figure1.png" alt="figure1" width="800"/>

We introduce FLeCS, a functional and learnable model of cell dynamics that incorporates gene network structure into 
coupled differential equations and: 
- accurately infers cell dynamics at scale
- provides improved functional insights into transcriptional mechanisms
- simulates single-cell trajectories

For a quick overview, please refer to `notebooks/Overview.ipynb` and `notebooks/SingleCellExample.ipynb`. 

## Installation
### Pip
You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install flecs:

<!--
1) Install the latest release of `flecs` from `PyPI <https://pypi.org/project/flecs/>`_:

```bash
pip install flecs
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/bertinus/flecs.git@main
```

### Conda

To install the conda environment on GPU, please run
```
conda env create -f environment_gpu.yml
conda activate flecs
pip install -e .
```

To install the conda environment on CPU, please run
```
conda env create -f environment_cpu.yml
conda activate flecs
pip install -e .
```

## Documentation

Please refer to the [documentation][link-docs].


[link-docs]: https://bertinus.github.io/FLeCS/
