class Dimension(object):
    def __init__(self, dim, selections=None):
        self._dim = dim
        self._type = self._get_type(dim, selections)

    @classmethod
    def _get_type(cls, dim, selections=None):
        '''Gets the Dimension type.

        MR and CA variables have two subsequent dimension, which are both
        necessary to determine the correct type ('categorical_array', or
        'multiple_response').
        '''
        type_ = dim['type'].get('class')

        if type_ and type_ == 'enum' and 'subreferences' in dim['references']:
            return ('multiple_response'
                    if cls._is_multiple_response(selections)
                    else 'categorical_array')

        if type_:
            return type_

        return dim['type']['subtype']['class']

    @classmethod
    def _is_multiple_response(cls, dim):
        if not dim:
            return False
        ids = [cat['id'] for cat in dim['type']['categories']]
        return ids == [1, 0, -1]

    @classmethod
    def _var_references(cls, dim):
        return {
            'refs': dim['references'],
            'type': cls._get_type(dim),
            'categories': dim['type']['categories'],
        }

    @classmethod
    def _get_name(cls, element):
        name = element.get('name')

        # For categorical variables
        if name:
            return name

        # For numerical, datetime and text variables
        value = element.get('value')
        if not value:
            return None

        # The following statement is used for compatibility between
        # python 2 and 3. In python 3 everything is 'str' and 'unicode'
        # is not defined. So, if the value is textual, in python 3 the first
        # part of the 'or' statement should short-circuit to 'True'.
        type_ = type(value)
        if not type_ == dict and (type_ == str or type_ == unicode):
            return value

        # For categorical array variables
        name = value.get('references', {}).get('name')
        if name:
            return name

        return None

    # API methods

    @property
    def type(self):
        return self._type

    def labels(self, include_missing=False):
        valid_indices = self.valid_indices(include_missing)
        return [
            self._get_name(el) for (i, el) in enumerate(self.elements)
            if i in valid_indices
        ]

    @property
    def elements(self):
        if self.type == 'categorical':
            return self._dim['type']['categories']
        return self._dim['type']['elements']

    def valid_indices(self, include_missing):
        if include_missing:
            return [i for (i, el) in enumerate(self.elements)]
        else:
            return [i for (i, el) in enumerate(self.elements)
                    if not el['missing']]
