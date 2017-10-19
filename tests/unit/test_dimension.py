from mock import Mock
from unittest import TestCase
from cr.cube.dimension import Dimension


class TestDimension(TestCase):
    # Test class method '_contains_type'
    def test_contains_type_does_not_contain_type(self):
        elements = [
            {'id': Mock()},
            {'id': Mock()},
            {'id': Mock()},
        ]
        type_ = Mock()
        expected = False
        actual = Dimension._contains_type(elements, type_)
        self.assertEqual(actual, expected)

    def test_contains_type_contains_type(self):
        type_ = Mock()
        elements = [
            {'id': Mock()},
            {'id': type_},
            {'id': Mock()},
        ]
        expected = True
        actual = Dimension._contains_type(elements, type_)
        self.assertEqual(actual, expected)

    # Test class method '_is_uniform'
    def test_is_uniform_true(self):
        data = {
            'references': {'uniform_basis': Mock()}
        }
        expected = True
        actual = Dimension._is_uniform(data)
        self.assertEqual(actual, expected)

    def test_is_uniform_false(self):
        data = {
            'references': {}
        }
        expected = False
        actual = Dimension._is_uniform(data)
        self.assertEqual(actual, expected)

    # Test class method '_is_selections'
    def test_is_selections_false_when_not_categorical(self):
        data = {
            'references': {},
            'type': {
                'class': Mock()
            }
        }
        expected = False
        actual = Dimension._is_selections(data)
        self.assertEqual(actual, expected)

    def test_is_selections_false_when_not_selection_ids(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': Mock()},
                    {'id': Mock()},
                ],
            }
        }
        expected = False
        actual = Dimension._is_selections(data)
        self.assertEqual(actual, expected)

    def test_is_selections_false_when_uniform(self):
        data = {
            'references': {
                'uniform_basis': Mock()
            },
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': 1},
                    {'id': 0},
                    {'id': -1},
                ],
            }
        }
        expected = False
        actual = Dimension._is_selections(data)
        self.assertEqual(actual, expected)

    def test_is_selections_true_when_equals_selection_ids(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': 1},
                    {'id': 0},
                    {'id': -1},
                ],
            }
        }
        expected = True
        actual = Dimension._is_selections(data)
        self.assertEqual(actual, expected)

    # Test class method '_is_categorical'
    def test_is_categorical_false_when_is_selections(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': 1},
                    {'id': 0},
                    {'id': -1},
                ],
            }
        }
        expected = False
        actual = Dimension._is_categorical(data)
        self.assertEqual(actual, expected)

    def test_is_categorical_true_when_uniform(self):
        data = {
            'references': {
                'uniform_basis': Mock(),
            },
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': 1},
                    {'id': 0},
                    {'id': -1},
                ],
            }
        }
        expected = True
        actual = Dimension._is_categorical(data)
        self.assertEqual(actual, expected)

    def test_is_categorical_true(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [],
            }
        }
        expected = True
        actual = Dimension._is_categorical(data)
        self.assertEqual(actual, expected)

    def test_is_categorical_false_when_not_categorical(self):
        data = {'type': {'class': Mock()}}
        expected = False
        actual = Dimension._is_categorical(data)
        self.assertEqual(actual, expected)

    # Test class method ''
    def test_is_multiple_response_false_when_not_enum(self):
        data = {'type': {'class': Mock()}}
        expected = False
        actual = Dimension._is_multiple_response(data)
        self.assertEqual(actual, expected)

    def test_is_multiple_response_false_when_has_no_mr(self):
        data = {
            'type': {
                'class': 'enum',
                'elements': [
                    {'id': Mock()},
                    {'id': Mock()},
                ],
            }
        }
        expected = False
        actual = Dimension._is_multiple_response(data)
        self.assertEqual(actual, expected)

    def test_is_multiple_response_true_when_has_mr(self):
        data = {
            'type': {
                'class': 'enum',
                'elements': [
                    {'id': Mock()},
                    {'id': -127},
                ],
            }
        }
        expected = True
        actual = Dimension._is_multiple_response(data)
        self.assertEqual(actual, expected)

    def test_get_data_type_categorical(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': Mock()},
                    {'id': Mock()},
                ],
            }
        }
        expected = 'categorical'
        actual = Dimension._get_data_type(data)
        self.assertEqual(actual, expected)

    def test_get_data_type_multiple_response(self):
        data = {
            'type': {
                'class': 'enum',
                'elements': [
                    {'id': Mock()},
                    {'id': -127},
                ],
            }
        }
        expected = 'multiple_response'
        actual = Dimension._get_data_type(data)
        self.assertEqual(actual, expected)

    def test_get_data_type_selections(self):
        data = {
            'references': {},
            'type': {
                'class': 'categorical',
                'categories': [
                    {'id': 1},
                    {'id': 0},
                    {'id': -1},
                ],
            }
        }
        expected = 'selections'
        actual = Dimension._get_data_type(data)
        self.assertEqual(actual, expected)

    def test_get_data_type_subtype(self):
        class_ = Mock()
        data = {
            'type': {
                'class': Mock(),
                'subtype': {
                    'class': class_
                }
            }
        }
        expected = class_
        actual = Dimension._get_data_type(data)
        self.assertEqual(actual, expected)

    def test_get_data_type_cannot_extract_type(self):
        data = {'type': {'class': Mock()}}
        with self.assertRaises(ValueError) as ctx:
            Dimension._get_data_type(data)
        error = 'Could not extract data type from: {}'.format(data)
        self.assertEqual(str(ctx.exception), error)
