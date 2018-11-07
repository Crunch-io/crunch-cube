#!/usr/bin/env python
# coding: utf-8

import os
import re

from setuptools import find_packages, setup


def ascii_bytes_from(path, *paths):
    """
    Return the ASCII characters in the file specified by *path* and *paths*.
    The file path is determined by concatenating *path* and any members of
    *paths* with a directory separator in between.
    """
    file_path = os.path.join(path, *paths)
    with open(file_path) as f:
        ascii_bytes = f.read()
    return ascii_bytes


# read required text from files
thisdir = os.path.dirname(__file__)
init_py = ascii_bytes_from(thisdir, 'src', 'cr', 'cube', '__init__.py')
readme = ascii_bytes_from(thisdir, 'README.md')

# Read the version from cr.cube.__version__ without importing the package
# (and thus attempting to import packages it depends on that may not be
# installed yet). This allows users to check installed version with
# `python -c 'from cr.cube import __version__; print(__version__)`
version = re.search("__version__ = '([^']+)'", init_py).group(1)

install_requires = [
    'scipy',
    'tabulate',
]

test_requires = [
    'pytest',
    'pytest-cov',
    'mock',
]

setup(
    name='cr.cube',
    version=version,
    description="Crunch.io Cube library",
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/Crunch-io/crunch-cube/',
    classifiers=[
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    author='Crunch.io',
    author_email='dev@crunch.io',
    license='MIT License',
    install_requires=install_requires,
    tests_require=test_requires,
    extras_require={
        'testing': test_requires,
    },
    packages=find_packages('src', exclude=['tests']),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={},
    namespace_packages=['cr'],
    zip_safe=True,
)
