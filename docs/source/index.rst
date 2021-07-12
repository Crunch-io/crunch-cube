
Welcome to Crunch Cube's documentation!
=======================================

Crunch Cube allows you to manipulate cube responses from the Crunch API using
Python. We'll refer to these cube responses as *cubes* in the subsequent
text. When used in conjunction with `pycrunch`, this library can unlock
powerful second-order analytics and visualizations.

A *cube* is obtained from the *Crunch.io* platform as a JSON response to
a specific *query* created by a user. The most common usage is to obtain the
following:

* Cross correlation between different variables
* Margins of the cross-tab *cube*
* Proportions of the cross-tab *cube* (e.g. proportions of each single
  element to the entire sample size)

Crunch Cube allows you to access these values from a cube response without
dealing with the complexities of the underlying JSON format.

The data in a cube is often best represented in a table-like format. For this
reason, many API methods return data as a `numpy.ndarray` object.


Installation
--------------

The `Crunch Cube package <https://pypi.org/project/cr.cube/>`_ can be installed via `pip install`::

  pip install cr.cube


A quick example
-----------------

After the `cr.cube` package has been successfully installed, the usage
is as simple as:

.. code-block:: python

    >>> from cr.cube.cube import Cube

    >>> ### Obtain the crunch cube JSON payload using app.crunch.io, pycrunch, rcrunch or scrunch
    >>> ### And store it in the 'cube_JSON_response' variable

    >>> cube = Cube(cube_JSON_response)
    >>> print(cube)
    Cube(name='MyCube', dimension_types='CAT x CAT')
    >>> cube.counts
    np.array([[1169, 547],
              [1473, 1261]])


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


.. image:: https://www.travis-ci.org/Crunch-io/crunch-cube.svg?branch=master
   :target: https://www.travis-ci.org/Crunch-io/crunch-cube

.. image:: https://coveralls.io/repos/github/Crunch-io/crunch-cube/badge.svg?branch=master
   :target: https://coveralls.io/github/Crunch-io/crunch-cube?branch=master

.. image:: https://readthedocs.org/projects/crunch-cube/badge/?version=latest
   :target: http://crunch-cube.readthedocs.io/en/latest/?badge=latest


.. toctree::
  :hidden:

  quickstart


.. toctree::
   :caption: API Reference
   :maxdepth: 5
   :hidden:

   cube
   cubepart
   dimension


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
