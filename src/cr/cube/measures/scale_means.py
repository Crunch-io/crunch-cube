'''Contains implementation of the Means service class.'''

from __future__ import division

import numpy as np

from ..utils import lazyproperty


class ScaleMeans(object):
    '''Implementation of the Means service.'''
    def __init__(self, slice_):
        self._slice = slice_

    @lazyproperty
    def data(self):
        '''Get the means calculation.'''
        means = []
        table = self._slice.as_array()
        products = self._inner_prods(table, self.values)

        for axis, product in enumerate(products):
            if product is None:
                means.append(product)
                continue

            # Calculate means
            num = np.sum(product[self.valid_indices(axis)], axis)
            den = np.sum(table[self.valid_indices(axis)], axis)
            mean = num / den
            if not isinstance(mean, np.ndarray):
                mean = np.array([mean])
            means.append(mean)
        return means

    @lazyproperty
    def values(self):
        '''Get num values for means calculation.'''
        return [
            (
                np.array(dim.values)
                if dim.values and any(~np.isnan(dim.values)) else
                None
            )
            for dim in self._slice.dimensions
        ]

    def valid_indices(self, axis):
        return [
            (
                ~np.isnan(np.array(dim.values))
                if dim.values and any(~np.isnan(dim.values)) and axis == i else
                slice(None)
            )
            for i, dim in enumerate(self._slice.dimensions)
        ]

    def _inner_prods(self, contents, values):
        products = []
        for i, numeric in enumerate(values):
            if numeric is None:
                products.append(numeric)
                continue
            inflate = self._slice.ndim > 1 and not i
            numeric = numeric[:, None] if inflate else numeric
            product = contents * numeric
            products.append(product)
        return products
