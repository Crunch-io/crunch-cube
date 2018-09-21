import numpy as np

from mock import Mock
from mock import patch
from unittest import TestCase
from cr.cube.dimension import Dimension
from cr.cube.subtotal import Subtotal


class TestDimension(TestCase):

    insertions_with_bad_data = [
        {
            u'anchor': 101,
            u'name': u'This is respondent ideology',
        },
        {
            u'anchor': 2,
            u'args': [1, 2],
            u'function': u'subtotal',
            u'name': u'Liberal net',
        },
        {
            u'anchor': 5,
            u'args': [5, 4],
            u'function': u'subtotal',
            u'name': u'Conservative net',
        },
        {
            u'anchor': 'fake anchor',
            u'args': ['fake_arg_1', 'fake_arg_2'],
            u'function': u'fake_fcn_name_not_subtotal',
            u'name': u'Fake Name',
        }
    ]

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

    def test_is_not_multiple_response(self):
        expected = False
        actual = Dimension._is_multiple_response({'type': {'fake': Mock()}})
        self.assertEqual(actual, expected)

    def test_get_name_from_element_name(self):
        name = Mock()
        expected = name
        actual = Dimension._get_name({'name': name})
        self.assertEqual(actual, expected)

    def test_get_name_from_element_list_vals(self):
        list_vals = [1.2, 3.4]
        expected = '-'.join(str(el) for el in list_vals)
        actual = Dimension._get_name({'value': list_vals})
        self.assertEqual(actual, expected)

    def test_get_name_from_element_numeric_value(self):
        num_val = 1.2
        expected = str(num_val)
        actual = Dimension._get_name({'value': num_val})
        self.assertEqual(actual, expected)

    def test_get_name_none(self):
        expected = None
        actual = Dimension._get_name({})
        self.assertEqual(actual, expected)

    def test_dimension_description(self):
        desc = Mock()
        dim = Dimension({'type': Mock(), 'references': {'description': desc}})
        expected = desc
        actual = dim.description
        self.assertEqual(actual, expected)

    @patch('cr.cube.dimension.Dimension._elements', [
        {'id': 1}, {'id': 2}, {'id': 5}, {'id': 4}
    ])
    @patch('cr.cube.subtotal.Subtotal._all_dim_ids', [1, 2, 4, 5])
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_hs_names_with_bad_data(self, mock_type):
        '''Test H&S names with bad input data.

        This test ensures that H&S functionality doesn't break if it encounters
        bad transformations data, as is possible with some of the leftovers in
        the variables.
        '''
        mock_type.return_value = None
        insertions_with_bad_data = [
            {
                u'anchor': 0,
                u'name': u'This is respondent ideology',
            },
            {
                u'anchor': 2,
                u'args': [1, 2],
                u'function': u'subtotal',
                u'name': u'Liberal net',
            },
            {
                u'anchor': 5,
                u'args': [5, 4],
                u'function': u'subtotal',
                u'name': u'Conservative net',
            }
        ]
        transform_data = {
            'references': {
                'view': {
                    'transform': {'insertions': insertions_with_bad_data}
                }
            }
        }
        dim = Dimension(transform_data)
        actual = dim.subtotals
        actual_anchors = [st.anchor for st in actual]
        self.assertEqual(actual_anchors, [2, 5])

    @patch('cr.cube.dimension.Dimension._elements', [
        {'id': 1}, {'id': 2}, {'id': 5}, {'id': 4}
    ])
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_hs_indices_with_bad_data(self, mock_type):
        '''Test H&S indices with bad input data.

        This test ensures that H&S functionality doesn't break if it encounters
        bad transformations data, as is possible with some of the leftovers in
        the variables.
        '''
        mock_type.return_value = None
        dim_data = {
            'references': {
                'view': {
                    'transform': {'insertions': self.insertions_with_bad_data}
                }
            }
        }
        dim = Dimension(dim_data)
        expected = [
            {'anchor_ind': 1, 'inds': [0, 1]},
            {'anchor_ind': 2, 'inds': [2, 3]}
        ]
        actual = dim.hs_indices
        self.assertEqual(actual, expected)

    @patch('cr.cube.dimension.Dimension._elements', [
        {'id': 1}, {'id': 2}, {'id': 5}, {'id': 4}
    ])
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_hs_indices_with_empty_indices(self, mock_type):

        mock_type.return_value = None
        dim_data = {
            'references': {
                'view': {
                    'transform': {'insertions': [{
                        "function": "subtotal",
                        "args": [
                            7,
                            8,
                            9,
                            10,
                            11
                        ],
                        "anchor": "bottom",
                        "name": "test subtotal"
                    }]}
                }
            }
        }
        dim = Dimension(dim_data)
        expected = []
        actual = dim.hs_indices
        self.assertEqual(actual, expected)

    # pylint: disable=protected-access, missing-docstring
    @patch('cr.cube.dimension.Dimension.elements')
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_subtotals(self, mock_type, mock_elements):
        mock_type.return_value = None
        mock_elements.return_value = [{'id': 1}, {'id': 5}]
        dim_data = {
            'references': {
                'view': {
                    'transform': {'insertions': self.insertions_with_bad_data}
                }
            }
        }
        dim = Dimension(dim_data)
        actual = dim.subtotals
        assert len(actual) == 2
        assert isinstance(actual[0], Subtotal)
        assert actual[0]._data == self.insertions_with_bad_data[1]
        assert isinstance(actual[1], Subtotal)
        assert actual[1]._data == self.insertions_with_bad_data[2]
        assert actual[0].anchor == 'bottom'
        assert actual[1].anchor == 5

    @patch('cr.cube.dimension.Dimension._elements', [
        {'id': 111}, {'id': 222}, {'id': 333}, {'id': 444}, {'id': 555}
    ])
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_inserted_hs_indices_and_order(self, mock_type):
        mock_type.return_value = None
        dim_data = {
            'references': {
                'view': {
                    'transform': {
                        'insertions': [
                            {
                                u'anchor': u'bottom',
                                u'args': [111],
                                u'function': u'subtotal',
                                u'name': u'bottoms up one',
                            },
                            {
                                u'anchor': u'bottom',
                                u'args': [222],
                                u'function': u'subtotal',
                                u'name': u'bottoms up two',
                            },
                            {
                                u'anchor': u'bottom',
                                u'args': [333],
                                u'function': u'subtotal',
                                u'name': u'bottoms up three',
                            },
                            {
                                u'anchor': u'top',
                                u'args': [444],
                                u'function': u'subtotal',
                                u'name': u'on top one',
                            },
                            {
                                u'anchor': u'top',
                                u'args': [555],
                                u'function': u'subtotal',
                                u'name': u'on top two',
                            },
                            {
                                u'anchor': 333,
                                u'args': [555],
                                u'function': u'subtotal',
                                u'name': u'in the middle one',
                            },
                            {
                                u'anchor': 333,
                                u'args': [555],
                                u'function': u'subtotal',
                                u'name': u'in the middle two',
                            }
                        ]
                    }
                }
            }
        }
        dim = Dimension(dim_data)
        self.assertEqual(dim.inserted_hs_indices, [0, 1, 5, 6, 9, 10, 11])
        labels = [
            u'on top one', u'on top two', None, None, None,
            u'in the middle one', u'in the middle two', None, None,
            u'bottoms up one', u'bottoms up two', u'bottoms up three'
        ]
        self.assertEqual(labels, dim.labels(include_transforms=True))

    @patch('cr.cube.dimension.Dimension._elements', [
        {'numeric_value': 1},
        {'numeric_value': 2, 'missing': False},
        {'numeric_value': 3, 'missing': True},
        {'numeric_value': None},
    ])
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_values(self, mock_type):
        dim = Dimension({})
        mock_type.return_value = None
        expected = [1, 2, np.nan]
        actual = dim.values
        self.assertEqual(actual, expected)

    @patch('cr.cube.dimension.Dimension.is_selections', True)
    @patch('cr.cube.dimension.Dimension._get_type')
    def test_hs_indices_for_mr(self, mock_type):
        dim = Dimension({})
        expected = []
        actual = dim.hs_indices
        assert actual == expected
