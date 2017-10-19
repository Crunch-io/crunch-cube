# coding: utf-8

MULTIPLE_RESPONSE_TYPE = -127
CATEGORICAL = 'categorical'
ENUM = 'enum'
SELECTION_TYPE_IDS = (1, 0, -1)


class Dimension(object):

    @classmethod
    def _contains_type(cls, elements, type_):
        return any(type_ == element['id'] for element in elements)

    @classmethod
    def _is_uniform(cls, data):
        return bool(data['references'].get('uniform_basis'))

    @classmethod
    def _is_selections(cls, data):
        data_type = data['type']

        if data_type['class'] != CATEGORICAL:
            return False

        category_ids = tuple(
            category['id'] for category in data_type['categories']
        )
        if category_ids == SELECTION_TYPE_IDS and not cls._is_uniform(data):
            return True

        return False

    @classmethod
    def _is_categorical(cls, data):
        data_class = data['type']['class']
        return (
            data_class == CATEGORICAL and
            not cls._is_selections(data)
        )

    @classmethod
    def _is_multiple_response(cls, data):
        data_type = data['type']
        data_class = data_type['class']
        return (
            data_class == ENUM and
            cls._contains_type(data_type['elements'], MULTIPLE_RESPONSE_TYPE)
        )

    @classmethod
    def _get_data_type(cls, data):
        if cls._is_selections(data):
            return 'selections'
        elif cls._is_categorical(data):
            return 'categorical'
        elif cls._is_multiple_response(data):
            return 'multiple_response'
        elif data['type'].get('subtype'):
            return data['type']['subtype']['class']
        raise ValueError('Could not extract data type from: {}'.format(data))
