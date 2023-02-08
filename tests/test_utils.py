from flecs.utils import get_project_root, set_seed


def test_utils():
    get_project_root()
    set_seed(0)


def test_package_has_version():
    flecs.__version__
