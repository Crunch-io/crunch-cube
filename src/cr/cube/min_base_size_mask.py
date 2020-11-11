# encoding: utf-8

"""MinBaseSize class."""

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty
from cr.cube.enums import DIMENSION_TYPE as DT


class MinBaseSizeMask(object):
    """Helper for deciding which rows/columns to suppress, based on min base size.

    If a base value, that is used when calculating percentages, is less than a given
    minimum base size, then all of the values obtained in such a way need to
    suppressed. We achieve this by generating a mask, based on row/column/table
    marginal values and the shape of the underlying slice.
    """

    def __init__(self, slice_, size, hs_dims=None, prune=False):
        self._slice = slice_
        self._size = size
        self._hs_dims = hs_dims
        self._prune = prune

    @lazyproperty
    def column_mask(self):
        """ndarray, True where column margin <= min_base_size, same shape as slice."""
        mask = self._slice.columns_base < self._size

        if self._slice.columns_base.shape == self._slice.shape:
            # If margin shape is the same as slice's (such as in a col margin for
            # MR x CAT), don't broadcast the mask to the array shape, since
            # they're already the same.
            return mask

        # If the row margin is a row vector - broadcast it's mask to the array shape
        return np.logical_or(np.zeros(self._slice.shape, dtype=bool), mask)

    @lazyproperty
    def row_mask(self):
        """ndarray, True where row margin <= min_base_size, same shape as slice."""
        mask = self._slice.rows_base < self._size

        if self._slice.rows_base.shape == self._slice.shape:
            # If margin shape is the same as slice's (such as in a row margin for
            # CAT x MR), don't broadcast the mask to the array shape, since
            # they're already the same.
            return mask

        # If the row margin is a column vector - broadcast it's mask to the array shape
        return np.logical_or(np.zeros(self._slice.shape, dtype=bool), mask[:, None])

    @lazyproperty
    def table_mask(self):
        """ndarray, True where table margin <= min_base_size, same shape as slice."""
        mask = self._slice.table_base < self._size

        if self._slice.table_base.shape == self._slice.shape:
            return mask

        if self._slice.dimension_types[0] == DT.MR:
            # If the margin is a column vector - broadcast it's mask to the array shape
            return np.logical_or(np.zeros(self._slice.shape, dtype=bool), mask[:, None])

        return np.logical_or(np.zeros(self._slice.shape, dtype=bool), mask)
