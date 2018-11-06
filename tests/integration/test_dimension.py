# encoding: utf-8

"""Integration test suite for the cr.cube.dimension module."""

import numpy as np
import pytest

from unittest import TestCase

from cr.cube.crunch_cube import CrunchCube
from cr.cube.dimension import AllDimensions, Dimension, _Subtotal
from cr.cube.enum import DIMENSION_TYPE as DT

from ..fixtures import CR  # ---mnemonic: CR = 'cube-response'---


class DescribeIntegratedAllDimensions(object):

    def it_resolves_the_type_of_each_dimension(self, type_fixture):
        dimension_dicts, expected_types = type_fixture
        all_dimensions = AllDimensions(dimension_dicts)

        dimension_types = tuple(d.dimension_type for d in all_dimensions)

        assert dimension_types == expected_types

    def it_provides_access_to_the_apparent_dimensions(self):
        dimension_dicts = CR.CA_X_MR_WEIGHTED_HS['result']['dimensions']
        all_dimensions = AllDimensions(dimension_dicts)

        apparent_dimension_types = tuple(
            d.dimension_type for d in all_dimensions.apparent_dimensions
        )

        assert apparent_dimension_types == (DT.CA, DT.CA_CAT, DT.MR)

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        (CR.CAT_X_CAT, (DT.CAT, DT.CAT)),
        (CR.CA_X_MR_WEIGHTED_HS, (DT.CA, DT.CA_CAT, DT.MR, DT.MR_CAT)),
    ])
    def type_fixture(self, request):
        cube_response, expected_types = request.param
        dimension_dicts = cube_response['result']['dimensions']
        return dimension_dicts, expected_types


class TestDimension(TestCase):

    def test_subtotals_indices_single_subtotal(self):
        dimension = CrunchCube(CR.ECON_BLAME_WITH_HS).dimensions[0]
        hs_indices = dimension.hs_indices
        self.assertEqual(hs_indices, ((1, (0, 1)),))

    def test_inserted_hs_indices_single_subtotal(self):
        dimension = CrunchCube(CR.ECON_BLAME_WITH_HS).dimensions[0]
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
        dimension = Dimension(dimension_dict, DT.CAT)

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
        dimension = Dimension(dimension_dict, DT.BINNED_NUMERIC)

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
        dimension = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING).dimensions[0]
        hs_indices = dimension.hs_indices
        self.assertEqual(hs_indices, ((1, (0, 1)), ('bottom', (3, 4))))

    def test_inserted_hs_indices_two_subtotals(self):
        dimension = CrunchCube(CR.ECON_BLAME_WITH_HS_MISSING).dimensions[0]
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
        dimension = Dimension(dimension_dict, DT.CAT)

        assert dimension.inserted_hs_indices == [0, 1, 5, 6, 9, 10, 11]
        assert dimension.labels(include_transforms=True) == [
            'on top one', 'on top two', '', '', '',
            'in the middle one', 'in the middle two', '', '',
            'bottoms up one', 'bottoms up two', 'bottoms up three'
        ]

    def test_has_transforms_false(self):
        dimension = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).dimensions[1]
        expected = False
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_has_transforms_true(self):
        dimension = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS).dimensions[0]
        expected = True
        actual = dimension.has_transforms
        self.assertEqual(actual, expected)

    def test_hs_indices_for_mr(self):
        dimension = CrunchCube(CR.MR_X_CAT_HS)._all_dimensions[1]
        hs_indices = dimension.hs_indices
        assert hs_indices == ()

    def test_hs_indices_with_bad_data(self):
        cube = CrunchCube(CR.CA_WITH_NETS)

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
        dimension = Dimension(dimension_dict, DT.CAT)

        hs_indices = dimension.hs_indices

        print('hs_indices == %s' % [hs_indices])
        assert hs_indices == ((1, (0, 1)), (2, (2, 3)))

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
        dimension = Dimension(dimension_dict, DT.CAT)

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
        dimension = Dimension(dimension_dict, DT.CAT)

        numeric_values = dimension.numeric_values

        self.assertEqual(numeric_values, (1, 2, np.nan, np.nan))
