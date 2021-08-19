# encoding: utf-8

"""MinBaseSize class."""

from cr.cube.util import lazyproperty


class MinBaseSizeMask:
    """Helper for deciding which rows/columns to suppress, based on min base size.

    If a base value, that is used when calculating percentages, is less than a given
    minimum base size, then all of the values obtained in such a way need to
    suppressed. We achieve this by generating a mask, based on row/column/table
    unweighted base values and the shape of the underlying slice.
    """

    def __init__(self, slice_, size, hs_dims=None, prune=False):
        self._slice = slice_
        self._size = size
        self._hs_dims = hs_dims
        self._prune = prune

    @lazyproperty
    def column_mask(self):
        """ndarray, True where column unweighted bases <= min_base_size"""
        return self._slice.column_unweighted_bases < self._size

    @lazyproperty
    def row_mask(self):
        """ndarray, True where row unweighted bases <= min_base_size"""
        return self._slice.row_unweighted_bases < self._size

    @lazyproperty
    def table_mask(self):
        """ndarray, True where table unweighted bases <= min_base_size"""
        return self._slice.table_unweighted_bases < self._size
