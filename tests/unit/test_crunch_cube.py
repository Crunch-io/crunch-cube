# encoding: utf-8

"""Unit test suite for the cr.cube.crunch_cube module."""

import numpy as np
import pytest

from unittest import TestCase

from cr.cube.crunch_cube import CrunchCube
from cr.cube.dimension import AllDimensions, _ApparentDimensions, Dimension
from cr.cube.enum import DIMENSION_TYPE as DT

from ..unitutil import (
    class_mock, instance_mock, method_mock, Mock, patch, property_mock
)


class DescribeCrunchCube(object):

    def it_provides_access_to_its_dimensions(
            self, _all_dimensions_prop_, all_dimensions_,
            apparent_dimensions_):
        _all_dimensions_prop_.return_value = all_dimensions_
        all_dimensions_.apparent_dimensions = apparent_dimensions_
        cube = CrunchCube({})

        dimensions = cube.dimensions

        assert dimensions is apparent_dimensions_

    def it_knows_the_types_of_its_dimension(self, request, dimensions_prop_):
        dimensions_prop_.return_value = tuple(
            instance_mock(
                request, Dimension, name='dim-%d' % idx, dimension_type=dt
            )
            for idx, dt in enumerate((DT.CAT, DT.CA_SUBVAR, DT.MR, DT.MR_CAT))
        )
        cube = CrunchCube({})

        dim_types = cube.dim_types

        assert dim_types == (DT.CAT, DT.CA_SUBVAR, DT.MR, DT.MR_CAT)

    def it_knows_when_it_contains_means_data(self, has_means_fixture):
        cube_response, expected_value = has_means_fixture
        cube = CrunchCube(cube_response)

        has_means = cube.has_means

        assert has_means is expected_value

    def it_knows_when_it_has_an_mr_dimension(
            self, has_mr_fixture, mr_dim_ind_prop_):
        mr_dim_indices, expected_value = has_mr_fixture
        mr_dim_ind_prop_.return_value = mr_dim_indices
        cube = CrunchCube({})

        has_mr = cube.has_mr

        assert has_mr is expected_value

    def it_can_adjust_an_axis_to_help(
            self, request, adjust_fixture, dimensions_prop_):
        dimension_types, axis_cases = adjust_fixture
        dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, dimension_type=dimension_type)
            for dimension_type in dimension_types
        )
        cube = CrunchCube({})

        for axis, expected_value in axis_cases:
            adjusted_axis = cube._adjust_axis(axis)
            assert adjusted_axis == expected_value

    def but_it_raises_on_disallowed_adjustment(self, _is_axis_allowed_):
        _is_axis_allowed_.return_value = False
        axis = 42
        cube = CrunchCube({})

        with pytest.raises(ValueError):
            cube._adjust_axis(axis)

        _is_axis_allowed_.assert_called_once_with(cube, axis)

    def it_provides_its_AllDimensions_collection_to_help(
            self, AllDimensions_, all_dimensions_):
        cube_response = {'result': {'dimensions': [{'d': 1}, {'d': 2}]}}
        AllDimensions_.return_value = all_dimensions_
        cube = CrunchCube(cube_response)

        all_dimensions = cube._all_dimensions

        AllDimensions_.assert_called_once_with([{'d': 1}, {'d': 2}])
        assert all_dimensions is all_dimensions_

    def it_knows_whether_an_axis_is_marginable_to_help(
            self, request, allowed_fixture, dimensions_prop_):
        dimension_types, axis_cases = allowed_fixture
        dimensions_prop_.return_value = tuple(
            instance_mock(request, Dimension, dimension_type=dimension_type)
            for dimension_type in dimension_types
        )
        cube = CrunchCube({})

        for axis, expected_value in axis_cases:
            axis_is_marginable = cube._is_axis_allowed(axis)
            assert axis_is_marginable is expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(params=[
        # ---0 - CAT x CAT---
        ((DT.CAT, DT.CAT),
         ((0, (0,)), (1, (1,)), (None, (0, 1)), ((0, 1), (0, 1)))),
        # ---1 - CAT x CAT x CAT---
        ((DT.CAT, DT.CAT, DT.CAT),
         ((0, (0,)), (1, (1,)), (2, (2,)), (None, (1, 2)), ((1, 2), (1, 2)))),
        # ---2 - MR (univariate)---
        ((DT.MR,),
         ((0, (1,)), (None, (1,)))),
        # ---3 - CAT x MR---
        ((DT.CAT, DT.MR),
         ((0, (0,)), (1, (2,)), (None, (0, 2)), ((0, 1), (0, 2)))),
        # ---4 - MR x CAT---
        ((DT.MR, DT.CAT),
         ((0, (1,)), (1, (2,)), (None, (1, 2)), ((0, 1), (1, 2)))),
        # ---5 - MR x MR---
        ((DT.MR, DT.MR),
         # --col---  --row----  --table-------------------------
         ((0, (1,)), (1, (3,)), (None, (1, 3)), ((0, 1), (1, 3)))),
        # ---6 - CAT x MR x MR---
        ((DT.CAT, DT.MR, DT.MR),
         # --0th---  --col----  --row----  --table-------------------------
         ((0, (0,)), (1, (2,)), (2, (4,)), (None, (2, 4)), ((1, 2), (2, 4)))),
        # ---7 - MR x CAT x MR---
        ((DT.MR, DT.CAT, DT.MR),
         # --0th---  --col----  --row----  --table-------------------------
         ((0, (1,)), (1, (2,)), (2, (4,)), (None, (2, 4)), ((1, 2), (2, 4)))),
        # ---8 - MR x MR x CAT---
        ((DT.MR, DT.MR, DT.CAT),
         # --0th---  --col----  --row----  --table-------------------------
         ((0, (1,)), (1, (3,)), (2, (4,)), (None, (3, 4)), ((1, 2), (3, 4)))),
        # ---9 - CA---
        ((DT.CA, DT.CAT),
         # --row-----
         ((1, (1,)),)),
        # ---10 - CA x CAT---
        ((DT.CA, DT.CAT, DT.CAT),
         # --row-----
         ((1, (1,)), (2, (2,)), (None, (1, 2)), ((1, 2), (1, 2)))),
        # ---11 - CAT x CA---
        ((DT.CAT, DT.CA, DT.CAT),
         # --row-----
         ((2, (2,)),)),
        # ---12 - CA x MR---
        ((DT.CA, DT.CAT, DT.MR),
         # --row-----
         ((1, (1,)), (2, (3,)), (None, (1, 3)), ((1, 2), (1, 3)))),
        # ---13 - MR x CAT x CA---
        ((DT.MR, DT.CA, DT.CAT),
         # --0th---  --row----
         ((0, (1,)), (2, (3,)))),
    ])
    def adjust_fixture(self, request):
        dimension_types, axis_cases = request.param
        return dimension_types, axis_cases

    @pytest.fixture(params=[
        # ---0 - CA---
        ((DT.CA, DT.CAT),
         (0, None, (0, 1))),
    ])
    def adjust_raises_fixture(self, request):
        dimension_types, axis_cases = request.param
        return dimension_types, axis_cases

    @pytest.fixture(params=[
        # ---0 - CA---
        ((DT.CA, DT.CAT),
         ((0, False), (1, True), (None, False))),
        # ---1 - CA x CAT---
        ((DT.CA, DT.CAT, DT.CAT),
         ((0, False), (1, True), (2, True), (None, True), ((1, 2), True))),
        # ---2 - CAT x CA-CAT---
        ((DT.CAT, DT.CA, DT.CAT),
         ((0, True), (1, False), (2, True), (None, False), ((1, 2), False))),
        # ---3 - MR x CA---
        ((DT.MR, DT.CA, DT.CAT),
         ((0, True), (1, False), (2, True), (None, False), ((1, 2), False))),
        # ---4 - CA x MR---
        ((DT.CA, DT.CAT, DT.MR),
         ((0, False), (1, True), (2, True), (None, True), ((1, 2), True))),
        # ---5 - Univariate CAT---
        ((DT.CAT,),
         ((0, True), (1, True), (None, True))),
        # ---6 - CAT x CAT---
        ((DT.CAT, DT.CAT),
         ((0, True), (1, True), (None, True), ((0, 1), True))),
        # ---7 - CAT x MR x MR---
        ((DT.CAT, DT.MR, DT.MR),
         ((0, True), (1, True), (2, True), (None, True), ((1, 2), True))),
        # ---8 - MR x CAT x MR---
        ((DT.MR, DT.CAT, DT.MR),
         ((0, True), (1, True), (2, True), (None, True), ((1, 2), True))),
        # ---9 - MR x MR x CAT---
        ((DT.MR, DT.MR, DT.CAT),
         ((0, True), (1, True), (2, True), (None, True), ((1, 2), True))),
    ])
    def allowed_fixture(self, request):
        dimension_types, axis_cases = request.param
        return dimension_types, axis_cases

    @pytest.fixture(params=[
        ({'result': {}}, False),
        ({'result': {'measures': {}}}, False),
        ({'result': {'measures': {'mean': {}}}}, True),
    ])
    def has_means_fixture(self, request):
        cube_response, expected_value = request.param
        return cube_response, expected_value

    @pytest.fixture(params=[
        (None, False),
        (0, True),
        (3, True),
        ([2, 5], True),
    ])
    def has_mr_fixture(self, request):
        mr_dim_indices, expected_value = request.param
        return mr_dim_indices, expected_value

    # fixture components ---------------------------------------------

    @pytest.fixture
    def AllDimensions_(self, request):
        return class_mock(request, 'cr.cube.crunch_cube.AllDimensions')

    @pytest.fixture
    def all_dimensions_(self, request):
        return instance_mock(request, AllDimensions)

    @pytest.fixture
    def _all_dimensions_prop_(self, request):
        return property_mock(request, CrunchCube, '_all_dimensions')

    @pytest.fixture
    def apparent_dimensions_(self, request):
        return instance_mock(request, _ApparentDimensions)

    @pytest.fixture
    def dimensions_prop_(self, request):
        return property_mock(request, CrunchCube, 'dimensions')

    @pytest.fixture
    def _is_axis_allowed_(self, request):
        return method_mock(request, CrunchCube, '_is_axis_allowed')

    @pytest.fixture
    def mr_dim_ind_prop_(self, request):
        return property_mock(request, CrunchCube, 'mr_dim_ind')


# pylint: disable=invalid-name, no-self-use, protected-access
@patch('cr.cube.crunch_cube.CrunchCube.get_slices', lambda x: None)
class TestCrunchCube(TestCase):
    '''Test class for the CrunchCube unit tests.

    This class also tests the functionality of private methods,
    not just the API ones.
    '''

    def test_init_raises_value_type_on_initialization(self):
        with self.assertRaises(TypeError) as ctx:
            CrunchCube(Mock())
        expected = (
            'Unsupported type provided: {}. '
            'A `cube` must be JSON or `dict`.'
        ).format(type(Mock()))
        self.assertEqual(str(ctx.exception), expected)

    @patch('cr.cube.crunch_cube.json.loads')
    def test_init_invokes_json_loads_for_string_input(self, mock_loads):
        mock_loads.return_value = {'value': {'result': {'dimensions': []}}}
        fake_json = 'fake cube json'
        CrunchCube(fake_json)
        mock_loads.assert_called_once_with(fake_json)

    def test_calculate_constraints_sum_axis_0(self):
        prop_table = np.array([
            [0.32, 0.21, 0.45],
            [0.12, 0.67, 0.73],
        ])
        prop_cols_margin = np.array([0.2, 0.3, 0.5])
        axis = 0
        expected = np.dot((prop_table * (1 - prop_table)), prop_cols_margin)
        actual = CrunchCube._calculate_constraints_sum(prop_table,
                                                       prop_cols_margin, axis)
        np.testing.assert_array_equal(actual, expected)

    def test_calculate_constraints_sum_axis_1(self):
        prop_table = np.array([
            [0.32, 0.21, 0.45],
            [0.12, 0.67, 0.73],
        ])
        prop_rows_margin = np.array([0.34, 0.66])
        axis = 1
        expected = np.dot(prop_rows_margin, (prop_table * (1 - prop_table)))
        actual = CrunchCube._calculate_constraints_sum(prop_table,
                                                       prop_rows_margin, axis)
        np.testing.assert_array_equal(actual, expected)

    def test_calculate_constraints_sum_raises_value_error_for_bad_axis(self):
        with self.assertRaises(ValueError):
            CrunchCube._calculate_constraints_sum(Mock(), Mock(), 2)

    def test_cube_counts(self):
        cube = CrunchCube({'result': {}})
        assert cube.counts == (None, None)

        fake_count = Mock()
        cube = CrunchCube({'result': {'unfiltered': fake_count}})
        assert cube.counts == (fake_count, None)

        cube = CrunchCube({'result': {'filtered': fake_count}})
        assert cube.counts == (None, fake_count)

        cube = CrunchCube(
            {'result': {'unfiltered': fake_count, 'filtered': fake_count}}
        )
        assert cube.counts == (fake_count, fake_count)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions', None)
    def test_name_with_no_dimensions(self):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        expected = None
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_name_with_one_dimension(self, mock_dims):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        mock_dims[0].name = 'test'
        expected = 'test'
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions', None)
    def test_description_with_no_dimensions(self):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        expected = None
        actual = cube.name
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_description_with_one_dimension(self, mock_dims):
        fake_cube = {}
        cube = CrunchCube(fake_cube)
        mock_dims[0].description = 'test'
        expected = 'test'
        actual = cube.description
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.has_means', False)
    def test_missing_when_there_are_none(self):
        fake_cube = {'result': {}}
        cube = CrunchCube(fake_cube)
        expected = None
        actual = cube.missing
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_subsequent(self):
        initial_indices = [[1, 2, 3]]
        insertion_index = 2
        expected = [[1, 2, 3, 4]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices,
            insertion_index,
            dimension
        )
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_with_gap(self):
        initial_indices = [[0, 1, 2, 5, 6]]
        insertion_index = 2
        expected = [[0, 1, 2, 3, 6, 7]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices,
            insertion_index,
            dimension
        )
        self.assertEqual(actual, expected)

    def test_fix_valid_indices_zero_position(self):
        initial_indices = [[0, 1, 2, 5, 6]]
        insertion_index = -1
        expected = [[0, 1, 2, 3, 6, 7]]
        dimension = 0
        actual = CrunchCube._fix_valid_indices(
            initial_indices,
            insertion_index,
            dimension
        )
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions', [])
    def test_does_not_have_description(self):
        expected = None
        actual = CrunchCube({}).description
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_has_description(self, mock_dims):
        dims = [Mock(), Mock()]
        mock_dims.__get__ = Mock(return_value=dims)
        expected = dims[0].description
        actual = CrunchCube({}).description
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.has_means', False)
    def test_missing(self):
        missing = Mock()
        fake_cube = {'result': {'missing': missing}}
        expected = missing
        actual = CrunchCube(fake_cube).missing
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.has_means', True)
    def test_missing_with_means(self):
        missing = Mock()
        fake_cube = {'result': {'measures': {'mean': {'n_missing': missing}}}}
        expected = missing
        actual = CrunchCube(fake_cube).missing
        self.assertEqual(actual, expected)

    def test_test_filter_annotation(self):
        mock_cube = {'filter_names': Mock()}
        expected = mock_cube['filter_names']
        actual = CrunchCube(mock_cube).filter_annotation
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.is_weighted', False)
    def test_n_unweighted_and_has_no_weight(self):
        unweighted_count = Mock()
        weighted_counts = Mock()
        actual = CrunchCube({
            'result': {
                'n': unweighted_count,
                'measures': {
                    'count': {
                        'data': weighted_counts,
                    },
                },
            },
        }).count(weighted=False)

        expected = unweighted_count
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.is_weighted', True)
    def test_n_unweighted_and_has_weight(self):
        unweighted_count = Mock()
        weighted_counts = Mock()
        actual = CrunchCube({
            'result': {
                'n': unweighted_count,
                'measures': {
                    'count': {
                        'data': weighted_counts,
                    },
                },
            },
        }).count(weighted=False)

        expected = unweighted_count
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.is_weighted', False)
    def test_n_weighted_and_has_no_weight(self):
        unweighted_count = Mock()
        weighted_counts = Mock()
        actual = CrunchCube({
            'result': {
                'n': unweighted_count,
                'measures': {
                    'count': {
                        'data': weighted_counts,
                    },
                },
            },
        }).count(weighted=True)

        expected = unweighted_count
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.is_weighted', True)
    def test_n_weighted_and_has_weight(self):
        unweighted_count = Mock()
        weighted_counts = [1, 2, 3, 4]
        actual = CrunchCube({
            'result': {
                'n': unweighted_count,
                'measures': {
                    'count': {
                        'data': weighted_counts,
                    },
                },
            },
        }).count(weighted=True)

        expected = sum(weighted_counts)
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.is_weighted', 'fake_val')
    def test_is_weighted_invoked(self):
        cube = CrunchCube({})
        actual = cube.is_weighted
        assert actual == 'fake_val'

    @patch('cr.cube.crunch_cube.CrunchCube.has_means', 'fake_val')
    def test_has_means_invoked(self):
        cube = CrunchCube({})
        actual = cube.has_means
        assert actual == 'fake_val'

    def test_margin_pruned_indices_without_insertions(self):
        table = np.array([0, 1, 0, 2, 3, 4])
        expected = np.array([True, False, True, False, False, False])
        actual = CrunchCube._margin_pruned_indices(table, [], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_without_insertions_with_nans(self):
        table = np.array([0, 1, 0, 2, 3, np.nan])
        expected = np.array([True, False, True, False, False, True])
        actual = CrunchCube._margin_pruned_indices(table, [], 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_with_insertions(self):
        table = np.array([0, 1, 0, 2, 3, 4])
        insertions = [0, 1]
        expected = np.array([False, False, True, False, False, False])

        actual = CrunchCube._margin_pruned_indices(table, insertions, 0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_pruned_indices_with_insertions_and_nans(self):
        table = np.array([0, 1, 0, 2, 3, np.nan])
        insertions = [0, 1]
        expected = np.array([False, False, True, False, False, True])

        actual = CrunchCube._margin_pruned_indices(table, insertions, 0)
        np.testing.assert_array_equal(actual, expected)

    @patch('numpy.array')
    @patch('cr.cube.crunch_cube.CrunchCube.inserted_hs_indices')
    @patch('cr.cube.crunch_cube.CrunchCube.ndim', 1)
    def test_inserted_inds(self, mock_inserted_hs_indices,
                           mock_np_array):
        expected = Mock()
        mock_np_array.return_value = expected

        cc = CrunchCube({})

        # Assert indices are not fetched without trasforms
        actual = cc._inserted_dim_inds(None, 0)
        # Assert np.array called with empty list as argument
        mock_np_array.assert_called_once_with([])
        assert actual == expected

        # Assert indices are fetch with transforms
        actual = cc._inserted_dim_inds([1], 0)
        mock_inserted_hs_indices.assert_not_called()
        actual = cc._inserted_dim_inds([0], 0)
        mock_inserted_hs_indices.assert_called_once()

    def test_population_fraction(self):
        # Assert fraction is 1 when none of the counts are specified
        cc = CrunchCube({})
        actual = cc.population_fraction
        assert actual == 1

        # Assert fraction is 1 when only some counts are specified
        cc = CrunchCube({'result': {'unfiltered': {'unweighted_n': 10}}})
        assert cc.population_fraction == 1
        cc = CrunchCube({'result': {'unfiltered': {'weighted_n': 10}}})
        assert cc.population_fraction == 1
        cc = CrunchCube({'result': {'unfiltered': {
            'weighted_n': 10, 'unweighted_n': 10}}})
        assert cc.population_fraction == 1
        cc = CrunchCube({'result': {'filtered': {
            'weighted_n': 10, 'unweighted_n': 10}}})
        assert cc.population_fraction == 1

        # Assert fraction is calculated when correct counts are specified
        cc = CrunchCube({
            'result': {
                'filtered': {'weighted_n': 5},
                'unfiltered': {'weighted_n': 10},
            }
        })
        assert cc.population_fraction == 0.5

        # Assert fraction is NaN, when denominator is zero
        cc = CrunchCube({
            'result': {
                'filtered': {'weighted_n': 5},
                'unfiltered': {'weighted_n': 0},
            }
        })
        assert np.isnan(cc.population_fraction)
