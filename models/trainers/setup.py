"""Using `setuptools` to create a source distribution."""

from setuptools import find_packages, setup

setup(
    name="mid-trainer",
    version="0.1",
    packages=['mid'],
    include_package_data=True,
    description="Midfielder model training application.",
)
