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
