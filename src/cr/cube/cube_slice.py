'''Home of the CubeSlice class.'''

from __future__ import division

import numpy as np


class CubeSlice(object):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    def __init__(self, cube, index):
        self._cube = cube
        self._index = index

    @property
    def ndim(self):
        '''Get number of dimensions.'''
        return self._cube.ndim

    @property
    def name(self):
        '''Get slice name.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        '''
        title = self._cube.name

        if self.ndim < 3:
            return title

        table_name = self._cube.labels()[0][self._index]
        return '%s: %s' % (title, table_name)

    @property
    def rows_title(self):
        '''Get title of the rows dimension.

        For 3D it's the 1st dimension (0th dimension of the current slice).
        '''
        return self._cube.dimensions[1].name

    @staticmethod
    def _increment_axis(axis):
        if axis is not None and not isinstance(axis, tuple):
            axis += 1
        # return (1, 2)
        return axis

    def as_array(self, *args, **kwargs):
        '''Call cube's as_array, and return correct slice.'''
        array = self._cube.as_array(*args, **kwargs)
        return array[self._index]

    def proportions(self, *args, **kwargs):
        '''Call cube's proportions, and return correct slice.'''
        kwargs['axis'] = self._increment_axis(kwargs.get('axis'))
        array = self._cube.proportions(*args, **kwargs)
        return array[self._index]

    def margin(self, *args, **kwargs):
        '''Call cube's margin, and return correct slice.'''
        kwargs['axis'] = self._increment_axis(kwargs.get('axis'))
        margin = self._cube.margin(*args, **kwargs)
        if len(margin) > 1:
            return margin[self._index]
        return margin

    @property
    def inserted_rows_indices(self):
        ''' Get correct inserted rows indices for the corresponding slice.

        For 3D cubes, a list of tuples is returned from the cube, when invoking
        the inserted_hs_indices method. The job of this property is to fetch
        the correct tuple (the one corresponding to the current slice index),
        and return the 0th value (the one corresponding to the rows).
        '''
        return self._cube.inserted_hs_indices()[self._index][0]

    def labels(self, *args, **kwargs):
        '''Return correct labels for slice.'''
        return self._cube.labels(*args, **kwargs)[-2:]

    def prune_indices(self, *args, **kwargs):
        '''Extract correct row/col prune indices from 3D cube.'''
        return list(self._cube.prune_indices(*args, **kwargs)[self._index])

    @property
    def has_means(self):
        '''Get has_means from cube.'''
        return self._cube.has_means

    @property
    def dimensions(self):
        return self._cube.dimensions[-2:]
