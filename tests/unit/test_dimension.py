from mock import Mock
from unittest import TestCase
from cr.cube.dimension import Dimension


class TestDimension(TestCase):
    def test_get_type_multiple_response(self):
        dim = {
            'references': {'subreferences': []},
            'type': {'class': 'enum'},
        }
        expected = 'multiple_response'
        actual = Dimension._get_type(dim)
        self.assertEqual(actual, expected)

    def test_get_type_categorical(self):
        dim = {
            'type': {'class': 'categorical'},
        }
        expected = 'categorical'
        actual = Dimension._get_type(dim)
        self.assertEqual(actual, expected)

    def test_get_type_numeric(self):
        dim = {
            'type': {'subtype': {'class': 'numeric'}},
        }
        expected = 'numeric'
        actual = Dimension._get_type(dim)
        self.assertEqual(actual, expected)

    def test_get_type_datetime(self):
        dim = {
            'type': {'subtype': {'class': 'datetime'}},
        }
        expected = 'datetime'
        actual = Dimension._get_type(dim)
        self.assertEqual(actual, expected)

    def test_get_type_text(self):
        dim = {
            'type': {'subtype': {'class': 'text'}},
        }
        expected = 'text'
        actual = Dimension._get_type(dim)
        self.assertEqual(actual, expected)
