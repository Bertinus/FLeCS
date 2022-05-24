from setuptools import setup, find_packages

setup(
    name="flecs",
    description="Flexible and Learnable Cell Simulator",
    packages=find_packages(include=["flecs", "flecs.*"]),
    version="0.1",
)
