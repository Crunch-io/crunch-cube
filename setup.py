#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

version = '1.6.7'


def get_long_desc():
    with open('README.md') as f:
        return f.read()


install_requires = [
    'scipy',
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
