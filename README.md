# crunch-cube

Open Source Python implementation of the API for working with CrunchCubes

## Introduction

This package contains the implementation of the CrunchCube API. It is used to
extract useful information from CrunchCube responses (we'll refer to them as
_cubes_ in the subsequent text). _Cubes_ are obtained from the *Crunch.io*
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

The `cr.cube` package can be installed by using the `pip install`:

    pip install cr.cube


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


    from cr.cube.crunch_cube import CrunchCube

    ### Obtain the crunch cube JSON from the Crunch.io
    ### And store it in the 'cube_JSON_response' variable

    cube = CrunchCube(cube_JSON_response)
    cube.as_array()

    ### Outputs:
    #
    # np.array([
    #     [5, 2],
    #     [5, 3]
    # ])

## API

### `as_array`

Tabular, or matrix, representation of the _cube_. The detailed description can
be found
[here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `margin`

Calculates margins of the _cube_. The detailed description can be found
[here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `proportions`

Calculates proportions of single variable elements to the whole sample size.
The detailed description can be found
[here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `percentages`

Calculates percentages of single variable elements to the whole sample size.
The detailed description can be found
[here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

[![Build Status](https://travis-ci.org/Crunch-io/crunch-cube.png?branch=master)](https://travis-ci.org/Crunch-io/crunch-cube)
[![Coverage Status](https://coveralls.io/repos/github/Crunch-io/crunch-cube/badge.svg?branch=master)](https://coveralls.io/github/Crunch-io/crunch-cube?branch=master)
[![Documentation Status](https://readthedocs.org/projects/crunch-cube/badge/?version=latest)](http://crunch-cube.readthedocs.io/en/latest/?badge=latest)


## Changes

### 1.12.1a
- Smoothing POC

### 1.11.37
- PR 216: Document matrix.py classes and properties

### 1.11.36
- Hypotesis testing for subtotals (heading and insertions)

### 1.11.35
- Bug fix for hypothesis testing with overlaps

### 1.11.34
- Bug fix for augmented MRxMR matrices

#### 1.11.33
- Manage augmentation for MRxMR matrices

#### 1.11.32
- Handle hidden option for insertions

#### 1.11.31
- Use bases instead of margin for MR `standard_error` calculation

#### 1.11.30
- Fix `standard_error` calculation for MR types 

#### 1.11.29
- Fix `standard_error` denominator for `Strand` 

#### 1.11.28
- Fix collapsed `scale-mean-pairwise-indices`

#### 1.11.27
- Standard deviation and standard error for `Strand`

#### 1.11.26
- Fix `pairwise_indices()` array collapse when all values empty

#### 1.11.25
- Expose two-level pairwise-t-test

#### 1.11.24
- Bug fix for scale_median calculation

#### 1.11.23
- Expose population fraction in cube partitions

#### 1.11.22
- Additional summary measures for scale (`std_dev`, `std_error`, `median`)

#### 1.11.21
- Fix slicing for CA + single col filter

#### 1.11.20
- Fix cube title payload discrepancy

#### 1.11.19
- Fix problem where pre-ordering anchor-idx was used for locating inserted subtotal vectors
- Enable handling of filter-only multitable-template placeholders.
- New measures: table and columns standard deviation and standard error

#### 1.11.18
- Fix wrong proportions and base values when explicit order is expressed

#### 1.11.17
- Fix incorrect means values after hiding

For a complete list of changes see [history](https://github.com/Crunch-io/crunch-cube/blob/master/HISTORY.md).
