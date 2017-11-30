from mock import Mock
from unittest import TestCase
from cr.cube.dimension import Dimension


class TestDimension(TestCase):
    def test_get_type_categorical_array(self):
        dim = {
            'references': {'subreferences': []},
            'type': {'class': 'enum'},
        }
        expected = 'categorical_array'
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
            'class': 'enum',
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

    def test_labels_for_categoricals(self):
        name_cat_1 = Mock()
        name_cat_2 = Mock()
        name_cat_3 = Mock()
        dim = {
            'type': {
                'class': 'categorical',
                'categories': [
                    {
                        'name': name_cat_1,
                        'missing': False,
                    },
                    {
                        'name': name_cat_2,
                        'missing': False,
                    },
                    {
                        'name': name_cat_3,
                        'missing': True,
                    },
                ]
            }
        }
        # Get only non-missing
        expected = [name_cat_1, name_cat_2]
        actual = Dimension(dim).labels()
        self.assertEqual(actual, expected)
        # Get all
        expected = [name_cat_1, name_cat_2, name_cat_3]
        actual = Dimension(dim).labels(include_missing=True)
        self.assertEqual(actual, expected)

    def test_labels_for_numericals(self):
        val_num_1 = 'fake val 1'
        val_num_2 = 'fake val 2'
        val_num_3 = {}
        dim = {
            'type': {
                "subtype": {
                    "class": "numeric"
                },
                "elements": [
                    {
                        "id": 0,
                        "value": val_num_1,
                        "missing": False,
                    },
                    {
                        "id": 1,
                        "value": val_num_2,
                        "missing": False,
                    },
                    {
                        "id": 2,
                        "value": val_num_3,
                        "missing": True,
                    }
                ],
            }
        }
        # Get only non-missing
        expected = [val_num_1, val_num_2]
        actual = Dimension(dim).labels()
        self.assertEqual(actual, expected)
        # Get all
        expected = [val_num_1, val_num_2, None]
        actual = Dimension(dim).labels(include_missing=True)
        self.assertEqual(actual, expected)
