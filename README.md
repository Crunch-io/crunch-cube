# crunch-cube

Open Source Python implementation of the API for working with Crunch Cubes

## Introduction

This package contains the implementation of the Crunch Cube API. It is used to
extract useful information from Crunch Cube responses (we'll refer to them as
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

The Crunch Cube package can be installed by using the `pip install`:

    pip install cr.cube


### For developers

For development mode, Crunch Cube needs to be installed from the local checkout
of the `crunch-cube` repository. Navigate to the top-level folder of the repo,
on the local file system, and run:

    python setup.py develop

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

### 1.0 Initial release

### 1.1 Fix stray ipdb.

### 1.2 Support exporter

### 1.3 Implement Headers & Subtotals

### 1.4 Update based on tabbook tests from `cr.lib`

#### 1.4.1 Update based on deck tests from `cr.server`

#### 1.4.2 Fix bugs discovered by first `cr.exporter` deploy to alpha

#### 1.4.3 Fix bug (exporting 2D crtab with H&S on row only)

#### 1.4.4 Implement obtaining labels with category ids (useful for H&S in exporter)

#### 1.4.5 Fix MR x MR proportions calculation

### 1.5.0 Start implementing index table functionality

#### 1.5.1 Implement index for MR x MR

#### 1.5.2 Fix bugs with `anchor: 0` for H&S

#### 1.5.3 Fix bugs with invalid input data for H&S

### 1.6.0 Z-Score and bug fixes.

#### 1.6.1 `standardized_residuals` are now included.

#### 1.6.2 support "Before" and "After" in variable transformations since they exist in zz9 data.

#### 1.6.4 Fixes for 3d Pruning.

#### 1.6.5 Fixes for Pruning and Headers and subtotals.
- Population size support.
- Fx various calculations in 3d cubes.

#### 1.6.6 Added support for CubeSlice, which always represents a
- 2D cube (even if they're the slices of a 3D cube).
- Various fixes for support of wide-export

#### 1.6.7 Population fraction
- Various bugfixes and optimizations.
- Add property `population_fraction`. This is needed for the exporter to be able to calculate the correct population counts, based on weighted/unweighted and filtered/unfiltered states of the cube.
- Apply newly added `population_fraction` to the calculation of `population_counts`.
- Modify API for `scale_means`. It now accepts additional parameters `hs_dims` (defaults to `None`) and `prune` (defaults to `False`). Also, the format of the return value is slightly different in nature. It is a list of lists of numpy arrrays. It functions like this:

    - The outermost list corresponds to cube slices. If cube.ndim < 3, then it's a single-element list
    - Inner lists have either 1 or 2 elements (if they're a 1D cube slice, or a 2D cube slice, respectively).
    - If there are scale means defined on the corresponding dimension of the cube slice, then the inner list element is a numpy array with scale means. If it doesn't have scale means defined (numeric values), then the element is `None`.

- Add property `ca_dim_ind` to `CubeSlice`.
- Add property `is_double_mr` to `CubeSlice` (which is needed since it differs from the interpretation of the cube. E.g. MR x CA x MR will render slices which are *not* double MRs).
- Add `shape`, `ndim`, and `scale_means` to `CubeSlice`, for accessibility.
- `index` now also operates on slices (no api change).

#### 1.6.8 Scale Means Marginal
- Add capability to calculate the scale means marginal. This is used when analysing a 2D cube, and obtaining a sort of a "scale mean _total_" for each of the variables constituting a cube.

#### 1.6.9 Bugfix
- When Categorical Array variable is selected in multitable export, and Scale Means is selected, the cube fails, because it tries to access the non-existing slice (the CA is only _interpreted_ as multiple slices in tabbooks). This fix makes sure that the export cube doesn't fail in such case.

#### 1.6.10 Fix README on pypi
