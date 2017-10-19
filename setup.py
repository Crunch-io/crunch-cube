#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

import os
thisdir = os.path.abspath(os.path.dirname(__file__))


version = '0.1'


def get_long_desc():
    root_dir = os.path.dirname(__file__)
    if not root_dir:
        root_dir = '.'
    return open(os.path.join(root_dir, 'README.md')).read()

install_requires = [
]

test_requires = [
    'py.test'
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
    license='Proprietary',
    install_requires=install_requires,
    tests_require=test_requires,
    extras_require={
        'testing': test_requires,
    },
    packages=find_packages(),
    namespace_packages=['cr'],
    include_package_data=True,
    package_data={},
    zip_safe=True,
)
