from setuptools import find_packages, setup

setup(
    name="flecs",
    description="Flexible and Learnable Cell Simulator",
    packages=find_packages(include=["flecs", "flecs.*"]),
    version="0.1",
)
