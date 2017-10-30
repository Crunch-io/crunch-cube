class Dimension(object):
    def __init__(self, dim):
        self._dim = dim

    @classmethod
    def _get_type(cls, dim):
        type_ = dim['type']['class']
        if type_ == 'enum' and 'subreferences' in dim['references']:
            return 'multiple_response'
        return type_

    @classmethod
    def _var_references(cls, dim):
        return {
            'refs': dim['references'],
            'type': cls._get_type(dim),
            'categories': dim['type']['categories'],
        }

    # API methods

    @property
    def type(self):
        return self._get_type(self._dim)

    @property
    def categories(self):
        return self._dim['type']['categories']

    @property
    def references(self):
        return self._dim['references']

    def valid_indices(self, include_missing):
        if include_missing:
            return [i for (i, cat) in enumerate(self.categories)]
        else:
            return [i for (i, cat) in enumerate(self.categories)
                    if not cat['missing']]
