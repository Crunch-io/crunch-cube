# crunch-cube

![Build Status](https://github.com/Crunch-io/crunch-cube/workflows/CI/badge.svg?branch=master)
![Coverage Status](https://codecov.io/gh/Crunch-io/crunch-cube/branch/master/graph/badge.svg?token=C6auKOj8tZ)
![Documentation Status](https://readthedocs.org/projects/crunch-cube/badge/?version=latest)

---
Open Source Python implementation of the API for working with CrunchCubes

## Introduction

This package contains the implementation of the CrunchCube API. It is used to
extract useful information from CrunchCube responses (we'll refer to them as
_cubes_ in the subsequent text). _Cubes_ are obtained from the _Crunch.io_
platform, as JSON responses to the specific _queries_ created by the user.
These queries specify which data the user wants to extract from the Crunch.io
system. The most common usage is to obtain the following:

- Cross correlation between different variable
- Margins of the cross tab _cube_
- Proportions of the cross tab _cube_ (e.g. proportions of each single element to the entire sample size)
- Percentages

When the data is obtained from the Crunch.io platform, it needs to be
interpreted to the form that's convenient for a user. The actual shape of the
_cube_ JSON contains many internal details, which are not of essence to the
end-user (but are still necessary for proper _cube_ functionality).

The job of this library is to provide a convenient API that handles those
intricacies, and enables the user to quickly and easily obtain (extract) the
relevant data from the _cube_. Such data is best represented in a table-like
format. For this reason, the most of the API functions return some form of the
`ndarray` type, from the `numpy` package. Each function is explained in greater
detail, uner its own section, under the API subsection of this document.

## Installation

The `cr-cube` package can be installed by using the `pip install`:

    pip install cr-cube

### For developers

For development mode, `cr.cube` needs to be installed from the local checkout
of the `crunch-cube` repository. It is strongly advised to use `virtualenv`.
Assuming you've created and activated a virtual environment `venv`, navigate
to the top-level folder of the repo, on the local file system, and run:

    pip install -e .

or

    python setup.py develop

### Running tests

To setup and run tests, you will need to install `cr.cube` as well as testing
dependencies. To do this, from the root directory, simply run:

    pip install -e .[testing]

And then tests can be run using `py.test` in the root directory:

    pytest

## Usage

After the `cr.cube` package has been successfully installed, the usage is as
simple as:

    >>> from cr.cube.cube import Cube

    >>> ### Obtain the crunch cube JSON payload using app.crunch.io, pycrunch, rcrunch or scrunch
    >>> ### And store it in the 'cube_JSON_response' variable

    >>> cube = Cube(cube_JSON_response)
    >>> print(cube)
    Cube(name='MyCube', dimension_types='CAT x CAT')
    >>> cube.counts
    np.array([[1169, 547],
              [1473, 1261]])

## Complete API Doc

Please visit <https://crunch-cube.readthedocs.io/en/latest> for the API reference.

---

## Changes

### 3.1.2

- Bug fix for scale mean diff rows and cols

### 3.1.1

- Bug fix for subtotal diff (wave difference) for categorical date dimensions

### 3.1.0

- Subtotal diff (wave difference) for categorical date dimensions

### 3.0.45

- Enumerator refactoring

### 3.0.44

- Bug fix median measure for exporter

### 3.0.43

- Median measure

For a complete list of changes see [history](https://github.com/Crunch-io/crunch-cube/blob/master/HISTORY.md).
