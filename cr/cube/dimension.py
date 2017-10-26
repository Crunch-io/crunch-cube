class Dimension(object):
    def __init__(self, dim):
        self._name = self._var_references(dim)

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
