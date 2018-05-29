'''Home of the CubeSlice class.'''


class CubeSlice(object):
    '''Implementation of CubeSlice class.

    The CubeSlice is used to uniformly represent all cubes as 2D tables.
    For 1D cubes, this is achieved by inflating. For 3D cubes, this is
    achieved by slicing.
    '''

    def __init__(self, cube, index):
        self._cube = cube
        self._index = index

    def _update_args(self, kwargs):

        if self.ndim < 3:
            # If cube is 2D it doesn't actually have slices (itself is a slice).
            # In this case we don't need to convert any arguments, but just
            # pass them to the underlying cube (which is the slice).
            return kwargs

        axis = kwargs.get('axis')
        if axis is None or isinstance(axis, tuple):
            # If no axis was passed, we don't need to update anything. If axis
            # was a tuple, it means that the caller was expecting a 3D cube.
            return kwargs

        kwargs['axis'] += 1
        return kwargs

    def _update_result(self, result):
        if self.ndim < 3 or len(result) - 1 < self._index:
            return result
        return result[self._index]

    def _call_cube_method(self, method, *args, **kwargs):
        kwargs = self._update_args(kwargs)
        result = getattr(self._cube, method)(*args, **kwargs)
        return self._update_result(result)

    # Properties

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

    @property
    def inserted_rows_indices(self):
        ''' Get correct inserted rows indices for the corresponding slice.

        For 3D cubes, a list of tuples is returned from the cube, when invoking
        the inserted_hs_indices method. The job of this property is to fetch
        the correct tuple (the one corresponding to the current slice index),
        and return the 0th value (the one corresponding to the rows).
        '''
        return self._cube.inserted_hs_indices()[self._index][0]

    @property
    def has_means(self):
        '''Get has_means from cube.'''
        return self._cube.has_means

    @property
    def dimensions(self):
        '''Get slice dimensions.

        For 2D cubes just get their dimensions. For 3D cubes, don't get the
        first dimension (only take slice dimensions).
        '''
        return self._cube.dimensions[-2:]

    # API Methods

    def labels(self, *args, **kwargs):
        '''Return correct labels for slice.'''
        return self._cube.labels(*args, **kwargs)[-2:]

    def prune_indices(self, *args, **kwargs):
        '''Extract correct row/col prune indices from 3D cube.'''
        if self.ndim < 3:
            return self._cube.prune_indices(*args, **kwargs)
        return list(self._cube.prune_indices(*args, **kwargs)[self._index])

    def as_array(self, *args, **kwargs):
        '''Call cube's as_array, and return correct slice.'''
        return self._call_cube_method('as_array', *args, **kwargs)

    def proportions(self, *args, **kwargs):
        '''Call cube's proportions, and return correct slice.'''
        return self._call_cube_method('proportions', *args, **kwargs)

    def margin(self, *args, **kwargs):
        '''Call cube's margin, and return correct slice.'''
        return self._call_cube_method('margin', *args, **kwargs)

    def population_counts(self, *args, **kwargs):
        '''Get population counts.'''
        return self._call_cube_method('population_counts', *args, **kwargs)

    @property
    def standardized_residuals(self):
        '''Get cube's standardized residuals.'''
        return self._cube.standardized_residuals

    def index(self, *args, **kwargs):
        '''Get index.'''
        return self._call_cube_method('index', *args, **kwargs)

    @property
    def dim_types(self):
        return self._cube.dim_types[-2:]

    @property
    def pvals(self):
        return self._cube.pvals

    def inserted_hs_indices(self, *args, **kwargs):
        return self._cube.inserted_hs_indices(*args, **kwargs)[-2:]
