'''Home of the CubeSlice class.'''

from __future__ import division


class CubeSlice(object):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    def __init__(self, cube, index=None):
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

    def as_array(self, *args, **kwargs):
        '''Call cube's as_array, and return correct slice.'''
        array = self._cube.as_array(*args, **kwargs)
        return array if self._index is None else array[self._index]

    @property
    def inserted_rows_indices(self):
        ''' Get correct inserted rows indices for the corresponding slice.

        For 3D cubes, a list of tuples is returned from the cube, when invoking
        the inserted_hs_indices method. The job of this property is to fetch
        the correct tuple (the one corresponding to the current slice index),
        and return the 0th value (the one corresponding to the rows).
        '''
        return self._cube.inserted_hs_indices()[self._index][0]
