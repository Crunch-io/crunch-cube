# encoding: utf-8

"""A matrix is the 2D cube-data partition used by a slice.

The `Assembler` object provides the external interface for this module. It name derives
from its role to "assemble" a finished 2D array ("matrix") for a particular measure from
the base measure values and inserted subtotals, to reorder the rows and columns
according to the dimension *order* transforms, and to hide rows and columns that are
either hidden by the user or "pruned" because they contain no observations.
"""

from cr.cube.util import lazyproperty


class Assembler(object):
    """Provides measure and margin methods for a cube-slice.

    An assembled matrix is a 2D ndarray reflecting all ordering, insertion, and hiding
    transforms applied to the dimensions. An assembled margin is often a 1D ndarray
    which is similarly formed from inserted values, ordered, and value hiding applied.

    `cube` is the `cr.cube.Cube` object containing the data for this matrix. Note that
    not all the data in `cube` will necessarily be used by this matrix. When `cube` is
    more than 2-dimensional, it is "sliced" and each slice gets its own matrix (and
    `_Slice` object).

    `dimensions` is a pair (2-tuple) of (rows_dimension, columns_dimension) Dimension
    objects. These are always the last two dimensions of `cube` but may and often do
    have transformations applied that are not present on the `cube` dimensions from
    which they derive.

    `slice_idx` is an int offset indicating which portion of `cube` data to use for this
    matrix. There is one slice for each element of the first cube dimension (the "table"
    dimension) when the cube has more than two dimensions.
    """

    def __init__(self, cube, dimensions, slice_idx):
        self._cube = cube
        self._dimensions = dimensions
        self._slice_idx = slice_idx

    @lazyproperty
    def unweighted_counts(self):
        """2D np.int64 ndarray of unweighted-count for each cell."""
        return self._assemble_matrix(
            _SumSubtotals.blocks(self._cube_result_matrix, "unweighted_counts")
        )

    def _assemble_matrix(self, blocks):
        """Return 2D ndarray matrix assembled from `blocks`.

        The assembled matrix includes inserted vectors (rows and columns), has hidden
        vectors removed, and is ordered by whatever sort method is applied in the
        dimension transforms.
        """
        # --- These are assembled into a single 2D array, and then rearranged based on
        # --- row and column orders. All insertion, ordering, and hiding transforms are
        # --- reflected in the row and column orders. They each include (negative)
        # --- insertion idxs, hidden and pruned vector indices have been removed, and
        # --- the ordering method has been applied to determine the sequence each idx
        # --- appears in. This directly produces a final array that is exactly the
        # --- desired output.
        raise NotImplementedError

    @lazyproperty
    def _cube_result_matrix(self):
        """_BaseCubeResultMatrix subclass object appropriate to this cube-slice.

        This matrix object encapsulates cube-result array parsing and MR multi-value
        differences and provides a foundational set of second-order analysis measure and
        margin arrays.
        """
        raise NotImplementedError


class _BaseSubtotals(object):
    """Base class for Subtotals objects."""

    @classmethod
    def blocks(cls, cube_result_matrix, measure_propname=None):
        """Return base, row and col insertion, and intersection matrices.

        These are in the form ready for assembly.
        """
        raise NotImplementedError


class _SumSubtotals(_BaseSubtotals):
    """Subtotal "blocks" created by np.sum() on addends, primarily counts."""


# === CUBE-RESULT MATRIX OBJECTS ===


class _BaseCubeResultMatrix(object):
    """Base class for all cube-result matrix (2D second-order analyzer) objects."""
