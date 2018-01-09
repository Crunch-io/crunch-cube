'''Contains implementation of the Subtotal class, for Crunch Cubes.'''


class Subtotal(object):
    '''Implementation of the Insertion class for Crunch Cubes.

    Contains all functionality necessary for retrieving the information
    for subtotals. This functionality is used in the context
    of headers and subtotals.
    '''
    def __init__(self, data):
        self._data = data

    @property
    def is_valid(self):
        '''Test if the subtotal data is valid.'''
        if isinstance(self._data, dict):
            required_keys = {'anchor', 'args', 'function', 'name'}
            has_keys = set(self._data.keys()) == required_keys
            if has_keys:
                return self._data['function'] == 'subtotal'
        return False

    @property
    def anchor(self):
        '''Get the anchor of the subtotal (if it's valid).'''
        if self.is_valid:
            return self._data['anchor']
        return None

    @property
    def args(self):
        '''Get H&S args.'''
        if self.is_valid:
            return self._data['args']
        return []

    @property
    def data(self):
        '''Get data in JSON format.'''
        return self._data
