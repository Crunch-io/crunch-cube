'''Contains implementation of the Means service class.'''

from __future__ import division

import numpy as np


class ScaleMeans(object):
    '''Implementation of the Means service.'''
    def __init__(self, cube):
        self._cube = cube

    @property
    def data(self):
        '''Get the means calculation.'''
        table = self._cube.as_array()
        contents = self._inner_prod(table, self.values)

        if self._cube.has_mr and not self._cube.is_double_mr:
            axis = 1 - self._cube.mr_dim_ind
            return np.sum(contents, axis) / np.sum(table, axis)

        if self.valid_inds.all():
            return np.sum(contents, self.axis) / self._cube.margin(self.axis)
        else:
            num = np.sum(contents[self.contents_inds], self.axis)
            den = np.sum(table[self.contents_inds], self.axis)
            return num / den

    @property
    def axis(self):
        '''Get axis for means calculation.'''
        axis = 0
        if self._cube.ca_dim_ind == 0 or self._cube.ca_dim_ind == 2:
            axis = 1
        elif len(self._cube.dimensions) > 2 and self._cube.ca_dim_ind == 1:
            axis = 2
        return axis

    @property
    def values(self):
        '''Get num values for means calculation.'''
        return np.array([
            dim.values for dim in self._cube.dimensions
            if dim.values and any(~np.isnan(dim.values))
        ][int(len(self._cube.dimensions) > 2)])

    @property
    def valid_inds(self):
        '''Valid indices for numerical values.'''
        return ~np.isnan(self.values)

    @property
    def contents_inds(self):
        '''Create contents selection indices based on valid num vals.'''
        return [
            slice(None) if i != self.axis else self.valid_inds
            for i in range(len(self._cube.as_array().shape))
        ]

    def _inner_prod(self, contents, values):
        if len(contents.shape) == 3 and self._cube.ca_dim_ind == 0:
            values = values[:, np.newaxis]
        try:
            return contents * values
        except:
            return contents * values[:, np.newaxis]
