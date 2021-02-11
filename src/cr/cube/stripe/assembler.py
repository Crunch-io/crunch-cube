# encoding: utf-8

"""The `StripeAssembler` object provides the external interface for this module.

Its name derives from its role to "assemble" a finished 1D array ("stripe") for a
particular measure from the base measure values and inserted subtotals, to reorder the
rows according to the dimension *order* transforms, and to hide rows that are either
hidden by the user or "pruned" because they contain no observations.
"""

from __future__ import division

from cr.cube.util import lazyproperty


class StripeAssembler(object):
    """Provides measures, marginals, and totals for a (1D) strand cube-slice.

    An assembled stripe measure is a 1D ndarray reflecting all ordering, insertion, and
    hiding transforms applied to the dimension. An assembled margin is often a scalar.

    `cube` is the `cr.cube.Cube` object containing the data for this matrix.

    `rows_dimension` is the Dimension object describing the stripe.

    `slice_idx` is an int offset indicating which portion of `cube` data to use for this
    stripe.
    """

    def __init__(self, cube, rows_dimension, ca_as_0th, slice_idx):
        self._cube = cube
        self._rows_dimension = rows_dimension
        self._ca_as_0th = ca_as_0th
        self._slice_idx = slice_idx

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted count for each row of stripe."""
        return self._assemble_vector(self._measures.unweighted_counts.blocks)

    def _assemble_vector(self, blocks):
        """Return 1D ndarray of base_vector with inserted subtotals, in order.

        `blocks` is a pair of two 1D arrays, first the base-values and then the subtotal
        values of the stripe vector. The returned array is sequenced in the computed
        row order including possibly removing hidden or pruned values.
        """
        raise NotImplementedError

    @lazyproperty
    def _measures(self):
        """StripeMeasures collection object for this stripe."""
        raise NotImplementedError
