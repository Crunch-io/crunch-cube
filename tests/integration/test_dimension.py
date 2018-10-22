# encoding: utf-8

"""Integration test suite for the cr.cube.dimension module."""

import numpy as np

from unittest import TestCase

from cr.cube.crunch_cube import CrunchCube
from cr.cube.dimension import Dimension, _Subtotal

from .fixtures import (
    CA_SUBVAR_HS_X_MR_X_CA_CAT, CA_WITH_NETS, ECON_BLAME_WITH_HS,
    ECON_BLAME_WITH_HS_MISSING, ECON_BLAME_X_IDEOLOGY_ROW_HS,
    LOGICAL_UNIVARIATE, LOGICAL_X_CAT, MR_X_CAT_HS
)


class TestDimension(TestCase):

    def test_dimension_type(self):
        cube = CrunchCube(CA_SUBVAR_HS_X_MR_X_CA_CAT)
        dimension_types = [d.dimension_type for d in cube.dimensions]
        assert dimension_types == [
            'categorical_array', 'multiple_response', 'categorical'
        ]

    def test_subtotals_indices_single_subtotal(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS).dimensions[0]
        hs_indices = dimension.hs_indices
        self.assertEqual(hs_indices, ((1, (0, 1)),))

    def test_inserted_hs_indices_single_subtotal(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS).dimensions[0]
        # It can be verified what the inserted indices are, by comparing
        # labels with/without transforms.
        expected = [2]
        actual = dimension.inserted_hs_indices
        assert actual == expected

    def test_labels_for_categoricals(self):
        dimension_dict = {
            'type': {
                'class': 'categorical',
                'categories': [
                    {
                        'id': 1,
                        'name': 'Cat',
                        'missing': False,
                    },
                    {
                        'id': 2,
                        'name': 'Mouse',
                        'missing': False,
                    },
                    {
                        'id': -1,
                        'name': 'Iguana',
                        'missing': True,
                    },
                ]
            }
        }
        dimension = Dimension(dimension_dict)

        # ---get only non-missing---
        labels = dimension.labels()
        assert labels == ['Cat', 'Mouse']

        # ---get all---
        labels = dimension.labels(include_missing=True)
        assert labels == ['Cat', 'Mouse', 'Iguana']

    def test_labels_for_numericals(self):
        dimension_dict = {
            'type': {
                'class': 'enum',
                'elements': [
                    {
                        'id': 0,
                        'value': 'smallish',
                        'missing': False
                    },
                    {
                        'id': 1,
                        'value': 'kinda big',
                        'missing': False
                    },
                    {
                        'id': 2,
                        'value': {},
                        'missing': True
                    }
                ],
                'subtype': {
                    'class': 'numeric'
                }
            }
        }
        dimension = Dimension(dimension_dict)

        # ---non-missing labels---
        labels = dimension.labels()
        assert labels == ['smallish', 'kinda big']

        # ---all labels, both valid and missing---
        labels = dimension.labels(include_missing=True)
        assert labels == ['smallish', 'kinda big', '']

        # ---all labels, both valid and missing---
        labels = dimension.labels(include_cat_ids=True)
        assert labels == [('smallish', 0), ('kinda big', 1)]

    def test_subtotals_indices_two_subtotals(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS_MISSING).dimensions[0]
        hs_indices = dimension.hs_indices
        self.assertEqual(hs_indices, ((1, (0, 1)), ('bottom', (3, 4))))

    def test_inserted_hs_indices_two_subtotals(self):
        dimension = CrunchCube(ECON_BLAME_WITH_HS_MISSING).dimensions[0]
        # It can be verified what the inserted indices are, by comparing
        # labels with/without transforms.
        expected = [2, 6]
        actual = dimension.inserted_hs_indices
        self.assertEqual(actual, expected)

    def test_inserted_hs_indices_order_and_labels(self):
        dimension_dict = {
            'references': {
                'view': {
                    'transform': {
                        'insertions': [
                            {
                                'anchor': 'bottom',
                                'args': [111],
                                'function': 'subtotal',
                                'name': 'bottoms up one',
                            },
                            {
                                'anchor': 'bottom',
                                'args': [222],
                                'function': 'subtotal',
                                'name': 'bottoms up two',
                            },
                            {
                                'anchor': 'bottom',
                                'args': [333],
                                'function': 'subtotal',
                                'name': 'bottoms up three',
                            },
                            {
                                'anchor': 'top',
                                'args': [444],
                                'function': 'subtotal',
                                'name': 'on top one',
                            },
                            {
                                'anchor': 'top',
                                'args': [555],
                                'function': 'subtotal',
                                'name': 'on top two',
                            },
                            {
                                'anchor': 333,
                                'args': [555],
                                'function': 'subtotal',
                                'name': 'in the middle one',
                            },
                            {
                                'anchor': 333,
                                'args': [555],
                                'function': 'subtotal',
                                'name': 'in the middle two',
                            }
                        ]
                    }
                }
            },
            'type': {
                "categories": [
                    {
                        "id": 111,
                    },
                    {
                        "id": 222,
                    },
                    {
                        "id": 333,
                    },
                    {
                        "id": 444,
                    },
                    {
                        "id": 555,
                    }
                ],
                "class": "categorical"
            }
        }
        dimension = Dimension(dimension_dict)

        assert dimension.inserted_hs_indices == [0, 1, 5, 6, 9, 10, 11]
        assert dimension.labels(include_transforms=True) == [
            'on top one', 'on top two', '', '', '',
            'in the middle one', 'in the middle two', '', '',
            'bottoms up one', 'bottoms up two', 'bottoms up three'
        ]

    def test_has_transforms_false(self):
        dimension = CrunchCube(
            ECON_BLAME_X_IDEOLOGY_ROW_HS
        ).dimensions[1]
        expected = False
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_has_transforms_true(self):
        dimension = CrunchCube(
            ECON_BLAME_X_IDEOLOGY_ROW_HS
        ).dimensions[0]
        expected = True
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_hs_indices_for_mr(self):
        dimension = CrunchCube(MR_X_CAT_HS).all_dimensions[1]
        hs_indices = dimension.hs_indices
        assert hs_indices == ()

    def test_hs_indices_with_bad_data(self):
        cube = CrunchCube(CA_WITH_NETS)

        subvar_dim = cube.dimensions[0]
        anchor_idxs = [anchor_idx for anchor_idx, _ in subvar_dim.hs_indices]
        assert anchor_idxs == ['bottom', 'bottom']

        cat_dim = cube.dimensions[1]
        anchor_idxs = [anchor_idx for anchor_idx, _ in cat_dim.hs_indices]
        assert anchor_idxs == ['bottom', 'bottom']

    def test_skips_bad_data_for_hs_indices(self):
        """Test H&S indices with bad input data.

        This test ensures that H&S functionality doesn't break if it
        encounters bad transformations data, as is possible with some of the
        leftovers in the variables.
        """
        dimension_dict = {
            'references': {
                'view': {
                    'transform': {
                        'insertions': [
                            {
                                'anchor': 101,
                                'name': 'This is respondent ideology',
                            },
                            {
                                'anchor': 2,
                                'args': [1, 2],
                                'function': 'subtotal',
                                'name': 'Liberal net',
                            },
                            {
                                'anchor': 5,
                                'args': [5, 4],
                                'function': 'subtotal',
                                'name': 'Conservative net',
                            },
                            {
                                'anchor': 'fake anchor',
                                'args': ['fake_arg_1', 'fake_arg_2'],
                                'function': 'fake_fcn_name_not_subtotal',
                                'name': 'Fake Name',
                            }
                        ]
                    }
                }
            },
            'type': {
                'categories': [
                    {
                        'numeric_value': 1,
                        'id': 1,
                        'name': 'President Obama',
                        'missing': False
                    },
                    {
                        'numeric_value': 2,
                        'id': 2,
                        'name': 'Republicans in Congress',
                        'missing': False
                    },
                    {
                        'numeric_value': 5,
                        'id': 5,
                        'name': 'Not sure',
                        'missing': False
                    },
                    {
                        'numeric_value': 4,
                        'id': 4,
                        'name': 'Neither',
                        'missing': False
                    }
                ],
                'class': 'categorical',
                'ordinal': False
            }
        }
        dimension = Dimension(dimension_dict)

        hs_indices = dimension.hs_indices

        print('hs_indices == %s' % [hs_indices])
        assert hs_indices == ((1, (0, 1)), (2, (2, 3)))

    def test_logical_univariate_dim(self):
        cube = CrunchCube(LOGICAL_UNIVARIATE)
        dimension = cube.dimensions[0]
        expected = 'categorical'
        actual = dimension.dimension_type
        self.assertEqual(expected, actual)
        self.assertFalse(dimension.is_mr_selections(cube.all_dimensions))

    def test_logical_x_cat_dims(self):
        cube = CrunchCube(LOGICAL_X_CAT)
        logical_dim = cube.dimensions[1]
        self.assertEqual(cube.dimensions[0].dimension_type, 'categorical')
        self.assertEqual(logical_dim.dimension_type, 'categorical')

        self.assertTrue(logical_dim.is_selections)
        self.assertFalse(logical_dim.is_mr_selections(cube.all_dimensions))

    def test_subtotals(self):
        dimension_dict = {
            'references': {
                'view': {
                    'transform': {
                        'insertions': [
                            {
                                'anchor': 101,
                                'name': 'This is respondent ideology',
                            },
                            {
                                'anchor': 2,
                                'args': [1, 2],
                                'function': 'subtotal',
                                'name': 'Liberal net',
                            },
                            {
                                'anchor': 5,
                                'args': [5, 4],
                                'function': 'subtotal',
                                'name': 'Conservative net',
                            },
                            {
                                'anchor': 'fake anchor',
                                'args': ['fake_arg_1', 'fake_arg_2'],
                                'function': 'fake_fcn_name_not_subtotal',
                                'name': 'Fake Name',
                            }
                        ]
                    }
                }
            },
            'type': {
                'categories': [
                    {'id': 1},
                    {'id': 5},
                    {'id': 8},
                    {'id': 9},
                    {'id': -1},
                ],
                'class': 'categorical'
            }
        }
        dimension = Dimension(dimension_dict)

        subtotals = dimension._subtotals

        assert len(subtotals) == 2

        subtotal = subtotals[0]
        assert isinstance(subtotal, _Subtotal)
        assert subtotal.anchor == 'bottom'
        assert subtotal.addend_ids == (1,)
        assert subtotal.addend_idxs == (0,)
        assert subtotal.label == 'Liberal net'

        subtotal = subtotals[1]
        assert isinstance(subtotal, _Subtotal)
        assert subtotal.anchor == 5
        assert subtotal.addend_ids == (5,)
        assert subtotal.addend_idxs == (1,)
        assert subtotal.label == 'Conservative net'

    def test_numeric_values(self):
        dimension_dict = {
            'type': {
                'categories': [
                    {'id': 42, 'missing': False, 'numeric_value': 1},
                    {'id': 43, 'missing': False, 'numeric_value': 2},
                    {'id': 44, 'missing': True, 'numeric_value': 3},
                    {'id': 45, 'missing': False, 'numeric_value': None},
                    {'id': 46, 'missing': False}
                ],
                'class': 'categorical'
            }
        }
        dimension = Dimension(dimension_dict)

        numeric_values = dimension.numeric_values

        self.assertEqual(numeric_values, (1, 2, np.nan, np.nan))
