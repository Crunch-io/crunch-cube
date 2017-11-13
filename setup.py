#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages
import os

thisdir = os.path.abspath(os.path.dirname(__file__))
version = '0.1'


def get_long_desc():
    return open(os.path.join(thisdir, 'README.md')).read()


install_requires = [
    'scipy',
]

test_requires = [
    'pytest',
    'pytest-cov',
]

setup(
    name='cr.cube',
    version=version,
    description="Crunch.io Cube library",
    long_description=get_long_desc(),
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
