'''Contains implementation of the Range service class.'''

import numpy as np


class Range(object):
    '''Implementation of the Range service class for Crunch Cubes.'''
    def __init__(self, cube, axis, sum_axis):
        self._cube = cube
        self._axis = axis
        self._sum_axis = sum_axis

    @property
    def cube(self):
        '''Get stored cube object.'''
        return self._cube

    @property
    def axis(self):
        '''Get axis property.'''
        return self._axis

    @property
    def sum_axis(self):
        '''Get sum axis property.'''
        return self._sum_axis

    @property
    def data(self):
        '''Finds the margin ranges for the given axes.'''

        margin = self._cube.margin(axis=self.axis)
        if margin.ndim == 1:
            margin = margin[np.newaxis, :].T

        mins = np.amin(margin, self.sum_axis)
        maxs = np.amax(margin, self.sum_axis)

        return [self._range_label(min_, max_) for min_, max_ in zip(mins, maxs)]

    @staticmethod
    def _range_label(min_, max_):
        if min_ == max_:
            return str(int(min_))
        return '-'.join([str(int(min_)), str(int(max_))])
