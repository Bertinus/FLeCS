# FLeCS

Flexible and Learnable Cell Simulator

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]


## Documentation

Please refer to the [documentation][link-docs].


## Installation
### Pip
You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install flecst:

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

To install the conda environment, please run
```
conda env create -f environment.yml
conda activate flecs
pip install -e .
```


## Information to edit the documentation

We use mkdocs. More information is available on [how to get started](https://www.mkdocs.org/getting-started/)
and how to [deploy the documentation](https://www.mkdocs.org/user-guide/deploying-your-docs/).

The most important commands are:
- ```mkdocs serve```: starts a local server to preview your documentation.
- ```mkdocs build```
- ```mkdocs gh-deploy```: builds the docs, then commits and pushes them to the *gh-pages* branch of the repository.


## Release notes

See the [changelog][changelog].


## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].


## Citation

> t.b.a


[badge-docs]: https://img.shields.io/readthedocs/flecs
[badge-tests]: https://img.shields.io/github/actions/workflow/status/bertinus/flecs/test.yaml?branch=main
[changelog]: https://flecst.readthedocs.io/latest/changelog.html
[issue-tracker]: https://github.com/bertinus/flecs/issues
[link-docs]: https://bertinus.github.io/FLeCS/
[link-tests]: flecs/actions/workflows/test.yml
[scverse-discourse]: https://discourse.scverse.org/