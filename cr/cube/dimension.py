class Dimension(object):
    def __init__(self, dim):
        self._dim = dim

    @classmethod
    def _get_type(cls, dim):
        type_ = dim['type'].get('class')

        if type_ and type_ == 'enum' and 'subreferences' in dim['references']:
            return 'multiple_response'

        if type_:
            return type_

        return dim['type']['subtype']['class']

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

        if name:
            return name

        name = element.get('value')
        if isinstance(name, (str, unicode)):
            return name

        return '<NA>'

    # API methods

    @property
    def type(self):
        return self._get_type(self._dim)

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
