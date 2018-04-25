.. Crunch Cube documentation master file, created by
   sphinx-quickstart on Fri Oct 20 07:37:21 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Crunch Cube's documentation!
=======================================

The purpose of this library is to make it much easier to manipulate cube responses that come from
the crunch api.  (we'll refer to them as *cubes* in the subsequent text). When used in conjunction
with pycrunch, this library can unlock powerful second-order analytics and visualizations.

*Cubes* are obtained from the *Crunch.io* platform, as JSON responses to the specific *queries* created by the user.
These queries specify which data the user wants to extract from the Crunch.io system.
The most common usage is to obtain the following:

- Cross correlation between different variable
- Margins of the cross tab *cube*
- Proportions of the cross tab *cube* (e.g. proportions of each single element to the entire sample size)
- Percentages

When the data is obtained from the Crunch.io platform, it needs to be interpreted to the form that's convenient for a user. The actual shape of the *cube* JSON contains many internal details, which are not of essence to the end-user (but are still necessary for proper *cube* functionality).

The job of this library is to provide a convenient API that handles those intricacies, and enables the user to quickly and easily obtain (extract) the relevant data from the *cube*. Such data is best represented in a table-like format. For this reason, the most of the API functions return some form of the `ndarray` type, from the `numpy` package. Each function is explained in greater detail, uner its own section, under the API subsection of this document.


Installation
--------------

The Crunch Cube package can be installed by using the `pip install`::

  pip install cr.cube


A quick example
-----------------

After the `cr.cube` package has been successfully installed, the usage
is as simple as:

.. code-block:: python

    >>> from cr.cube.crunch_cube import CrunchCube

    >>> ### Obtain the crunch cube JSON from the Crunch.io
    >>> ### And store it in the 'cube_JSON_response' variable

    >>> cube = CrunchCube(cube_JSON_response)
    >>> cube.as_array()
    np.array([
         [5, 2],
         [5, 3]
     ])



For developers
---------------

For development mode, Crunch Cube needs to be installed from the local
checkout of the `crunch-cube` repository.  Navigate to the top-level
folder of the repo, on the local file system, and run::

  $ python setup.py develop
  $ py.test tests -cov=cr.cube

Note that we are happy to accept pull requests, please be certain that
your code has proper coverage before submitting.  All pull requests
will be tested by travis.


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   cr.cube


.. image:: https://www.travis-ci.org/Crunch-io/crunch-cube.svg?branch=master
    :target: https://www.travis-ci.org/Crunch-io/crunch-cube

.. image:: https://coveralls.io/repos/github/Crunch-io/crunch-cube/badge.svg?branch=master
    :target: https://coveralls.io/github/Crunch-io/crunch-cube?branch=master

.. image:: https://readthedocs.org/projects/crunch-cube/badge/?version=latest
    :target: http://crunch-cube.readthedocs.io/en/latest/?badge=latest



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
