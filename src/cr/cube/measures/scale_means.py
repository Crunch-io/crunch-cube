# encoding: utf-8

"""Objects related to "scale-mean" features.

Categorical variables (and perhaps others) allow a *numeric-value* to be
assigned to each category. This could be something like "like=1, dislike=-1,
neutral=0", or possibly "one-child=1, two-children=2, three-or-more=3".

These numeric values impose a quantitative *scale* on an otherwise
non-numeric variable, and thereby allow quantitative operations on that
variable.

Taking the mean value is one of those quantitative operations, and
"scale-mean" refers to calculating the *mean* value of the values on this
numeric *scale*.
"""

from __future__ import division

import numpy as np

from cr.cube.util import lazyproperty


class ScaleMeans(object):
    """Value object providing "mean-of-scale" values on a CubeSlice object."""

    def __init__(self, slice_):
        self._slice = slice_

    @lazyproperty
    def data(self):
        """list of mean numeric values of categorical responses."""
        means = []
        table = self._slice.as_array()
        products = self._inner_prods(table, self.values)

        for axis, product in enumerate(products):
            if product is None:
                means.append(product)
                continue

            # Calculate means
            num = np.sum(product[self._valid_indices(axis)], axis)
            den = np.sum(table[self._valid_indices(axis)], axis)
            mean = num / den
            if not isinstance(mean, np.ndarray):
                mean = np.array([mean])
            means.append(mean)
        return means

    def margin(self, axis):
        '''Return marginal value of the current slice scaled means.

        This value is the the same what you would get from a single variable
        (constituting a 2D cube/slice), when the "non-missing" filter of the
        opposite variable would be applied. This behavior is consistent with
        what is visible in the front-end client.
        '''
        if self._slice.ndim < 2:
            msg = (
                'Scale Means marginal cannot be calculated on 1D cubes, as'
                'the scale means already get reduced to a scalar value.'
            )
            raise ValueError(msg)

        dimension_index = 1 - axis
        margin = self._slice.margin(axis)
        total = np.sum(margin)
        values = self.values[dimension_index]

        if values is None:
            return None

        return np.sum(values * margin) / total

    def _valid_indices(self, axis):
        # --there is an interaction with CrunchCube._fix_shape() which
        # --essentially eliminates length-1 dimensions not in the first
        # --position. We must mirror that reshaping here. The fact this logic
        # --needs to be duplicated indicates we're missing an abstraction
        # --somewhere, like perhaps CrunchCube and/or
        # --CrunchSlice.reshaped_dimensions.
        reshaped_dimensions = [
            dimension for dimension in self._slice.dimensions
            if len(dimension.numeric_values) > 1
        ]

        return tuple(
            (
                ~np.isnan(np.array(dim.numeric_values))
                if (
                    dim.numeric_values and
                    any(~np.isnan(dim.numeric_values)) and
                    axis == i
                )
                else slice(None)
            )
            for i, dim in enumerate(reshaped_dimensions)
        )

    @lazyproperty
    def values(self):
        """list of ndarray value-ids for each dimension in slice.

        The values for each dimension appear as an ndarray. None appears
        instead of the array for each dimension having only NaN values.
        """
        return [
            (
                np.array(dim.numeric_values)
                if (
                    dim.numeric_values and
                    any(~np.isnan(dim.numeric_values))
                )
                else None
            )
            for dim in self._slice.dimensions
        ]

    def _inflate(self, dim_ind):
        return (
            self._slice.ndim > 1 and
            len(self._slice.get_shape()) > 1 and
            not dim_ind
        )

    def _inner_prods(self, contents, values):
        products = []
        for i, numeric in enumerate(values):
            if numeric is None:
                products.append(numeric)
                continue
            numeric = numeric[:, None] if self._inflate(i) else numeric
            product = contents * numeric
            products.append(product)
        return products
