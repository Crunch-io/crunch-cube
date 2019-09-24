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

    pip install -e ".[testing]"

And then tests can be run using `py.test` in the root directory:

    pytest

## Usage

After the `cr.cube` package has been successfully installed, the usage is as
simple as:


    from cr.cube.cube import Cube

    ### Obtain the crunch cube JSON from the Crunch.io
    ### And store it in the 'cube_JSON_response' variable

    cube = Cube(cube_JSON_response)
    slice_ = cube.partitions[0]
    print(slice_.column_proportions)

    ### Output (2D Cube):
    #
    # np.ndarray([
    # [0.38227848 0.33855186 0.39252336 0.44117647]
    # [0.09620253 0.10665362 0.18691589 0.08823529]
    # [0.47848101 0.44520548 0.57943925 0.52941176]
    # [0.         0.         0.         0.        ]
    # [0.09113924 0.09099804 0.14018692 0.17647059]
    # [0.25316456 0.23091977 0.08411215 0.05882353]
    # [0.09113924 0.13111546 0.13084112 0.14705882]
    # [0.07594937 0.07632094 0.05607477 0.08823529]
    # [0.01012658 0.02544031 0.00934579 0.        ]
    # ])
    
    print(slice_.row_proportions)
    
    ### Output (2D Cube:
    #
    # np.ndarray([
    # [0.27256318 0.62454874 0.07581227 0.02707581]
    # [0.22352941 0.64117647 0.11764706 0.01764706]
    # [0.26104972 0.62845304 0.08563536 0.02486188]
    # [       nan        nan        nan        nan]
    # [0.24       0.62       0.1        0.04      ]
    # [0.28818444 0.68011527 0.0259366  0.00576369]
    # [0.19047619 0.70899471 0.07407407 0.02645503]
    # [0.25641026 0.66666667 0.05128205 0.02564103]
    # [0.12903226 0.83870968 0.03225806 0.        ]
    # ])
    
    # How to get the data values (%) from the cube
    
    print(slice_.column_percentages)
    
    ### Output:
    #
    # np.ndarray([
    # [38.2278481 , 33.85518591, 39.25233645, 44.11764706],
    # [ 9.62025316, 10.66536204, 18.69158879,  8.82352941],
    # [47.84810127, 44.52054795, 57.94392523, 52.94117647],
    # [ 0.        ,  0.        ,  0.        ,  0.        ],
    # [ 9.11392405,  9.09980431, 14.01869159, 17.64705882],
    # [25.3164557 , 23.09197652,  8.41121495,  5.88235294],
    # [ 9.11392405, 13.11154599, 13.08411215, 14.70588235],
    # [ 7.59493671,  7.63209393,  5.60747664,  8.82352941],
    # [ 1.01265823,  2.54403131,  0.93457944,  0.        ]
    # ]) 
    
    print(cube.base_counts)
    
    ### Output:
    #
    # np.ndarray([
    # [151 346  42  15]
    # [ 38 109  20   3]
    # [  0   0   0   0]
    # [ 36  93  15   6]
    # [100 236   9   2]
    # [ 36 134  14   5]
    # [ 30  78   6   3]
    # [  4  26   1   0]
    # ])
    
## API

### `base_counts` or `counts`

`cube.counts` `cube.base_counts`

Tabular, or matrix, representation of the _cube_. 

[^Comment]: [here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `rows_margin`
`slice_.rows_margin`

Calculates rows margin of the _slice_.

[^Comment]: [here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `row_proportions` and `column_proportions`
`slice_.row_proportions` `slice_.column_proportions`

Calculates proportions of single variable elements to the whole sample size.

[^Comment]: [here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

### `row_percentages` and `column_percentages`

`slice_.row_percentages` `slice_.column_percentages`

Calculates percentages of single variable elements to the whole sample size.

[^Comment]: [here](http://crunch-cube.readthedocs.io/en/latest/cr.cube.html#cr-cube-crunch-cube-module).

[![Build Status](https://travis-ci.org/Crunch-io/crunch-cube.png?branch=master)](https://travis-ci.org/Crunch-io/crunch-cube)
[![Coverage Status](https://coveralls.io/repos/github/Crunch-io/crunch-cube/badge.svg?branch=master)](https://coveralls.io/github/Crunch-io/crunch-cube?branch=master)
[![Documentation Status](https://readthedocs.org/projects/crunch-cube/badge/?version=latest)](http://crunch-cube.readthedocs.io/en/latest/?badge=latest)


## Changes

#### 1.11.7
- Fix a bug when MR x MR table is pruned on both dimensions

#### 1.11.6
- Calculate population size fraction using complete cases

For a complete list of changes see [history](https://github.com/Crunch-io/crunch-cube/blob/master/HISTORY.md).
