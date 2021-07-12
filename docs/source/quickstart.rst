Quick Start
===========
In the Crunch system, any analysis is also referred to as a ``cube``. Cubes are
the mechanical means of representing analyses to and from the Crunch system;
you can think of them as spreadsheets that might have other than two dimensions.
A cube consists of two primary parts: “dimensions” which supply the cube axes,
and “measures” which populate the cells. Although both the request and response
include dimensions and measures, it is important to distinguish between them.
The request supplies expressions for each, while the response has data
(and metadata) for each. The request declares what variables to use and what
to do with them, while the response includes and describes the results.

At an abstract level, cubes contain arrays (``numpy arrays``) of measures.
Measures frequently (although not always!) are simply counts of responses that
fall into each cell of the cross-tabulation (also sometimes called contingency tables).
Cubes always include the unweighted counts which are important for some analyses,
or could contain other measures which are treated differently.

Check out the details `here <https://help.crunch.io/hc/en-us/articles/360044737751-Multidimensional-Analysis>`_

Installation
------------
The `Crunch Cube package <https://pypi.org/project/cr.cube/>`_ can be installed via `pip install`::

  pip install cr.cube

Cube object
-----------
Below a quick example on how instanciate and query the counts of a `cube`

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

If the JSON response includes both ``weighted`` and ``unweighted_counts``, ``cube.counts``
corresponds to the weighted version of the counts; but we still have both measures:

.. code-block:: python

    >>> cube.counts
    np.array([[1122.345, 234.456,
             1432.2331, 1211.8763]])
    >>> cube.unweighted_counts
    np.array([[1169, 547],
              [1473, 1261]])

Cube Partitions
---------------
A ``cube`` can contain 1 or more partitions according to its dimensionality.
For example a CAT_X_CAT cube has a single 2D partition, identified as a `Slice`
object in the `cubepart` module, a CA_SUBVAR_X_CA_CAT cube has two 2D partitions
that can be represented like:

.. code-block:: python

    >>> cube.partitions[0]
    _Slice(name='pets_array', dimension_types='CA_SUBVAR x CA_CAT')
    Showing: COUNT
              not selected    selected
    ------  --------------  ----------
    cat                 13          12
    dog                 16          12
    wombat              11          12
    Available measures: [<CUBE_MEASURE.COUNT: 'count'>]
    >>> cube.partitions[1]
    _Slice(name='pets_array', dimension_types='CA_SUBVAR x CA_CAT')
    Showing: COUNT
              not selected    selected
    ------  --------------  ----------
    cat                 32          22
    dog                 24          28
    wombat              21          26
    Available measures: [<CUBE_MEASURE.COUNT: 'count'>]

Let's back to the CAT_X_CAT cube, the example below shows how to access to some
of the avilable measures for the analyses.

.. code-block:: python

    >>> cube = Cube(cube_JSON_response_CAT_X_CAT)
    >>> partition = cube.partition[0]
    >>> partition.column_proportions
    array([[0.5, 0.4],
           [0.5, 0.6]])
    >>> partition.column_std_dev
    array([[0.5       , 0.48989795],
           [0.5       , 0.48989795]])
    >>> partition.columns_scale_mean
    array([1.5, 1.6])

For the complete measure references visit the `Partition API <cubepart.html>`_
