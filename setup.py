"""Using `setuptools` to create a source distribution."""

from setuptools import find_packages, setup

setup(
    name="mid_trainer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    description="Midfielder model training application.",
)
