'''Home of the CubeSlice class.'''

from __future__ import division

from .crunch_cube import CrunchCube


class CubeSlice(CrunchCube):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    def __init__(self, response, index=None):
        super(CubeSlice, self).__init__(response)
        self._index = index
