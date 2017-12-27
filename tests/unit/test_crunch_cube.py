'''Unit tests for the CrunchCube class.'''
from unittest import TestCase
from mock import Mock
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube


# pylint: disable=invalid-name
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

    @patch('cr.cube.crunch_cube.CrunchCube._has_means', False)
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

    @patch('cr.cube.crunch_cube.CrunchCube.has_multiple_response', False)
    @patch('cr.cube.crunch_cube.CrunchCube.as_array')
    @patch('cr.cube.crunch_cube.CrunchCube._is_double_multiple_response')
    @patch('cr.cube.crunch_cube.CrunchCube._margin')
    def test_transform_param_propagation(self, mock_margin,
                                         mock_is_double_mr, mock_as_array):
        mock_margin.return_value = 1  # Prevent 'proportions' from crashing
        mock_is_double_mr.return_value = False
        mock_as_array.return_value = 0
        cube = CrunchCube({})

        # Parameter: 'include_transforms_for_dims'
        fake_dims = Mock()
        fake_axis = Mock()
        fake_weighted = Mock()
        # Make the call
        cube.proportions(
            axis=fake_axis,
            weighted=fake_weighted,
            include_transforms_for_dims=fake_dims
        )
        # Assert parameter propagation
        mock_margin.assert_called_once_with(
            axis=fake_axis,
            weighted=fake_weighted,
            adjusted=False,
            include_transforms_for_dims=fake_dims,
        )

    @patch('cr.cube.crunch_cube.CrunchCube._get_dimensions')
    @patch('cr.cube.crunch_cube.CrunchCube._get_mr_selections_indices')
    def test_does_not_have_multiple_response(self, mock_mr_indices, mock_dims):
        mock_mr_indices.return_value = []
        expected = False
        actual = CrunchCube({}).has_multiple_response
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube._get_dimensions')
    @patch('cr.cube.crunch_cube.CrunchCube._get_mr_selections_indices')
    def test_has_multiple_response(self, mock_mr_indices, mock_dims):
        mock_mr_indices.return_value = [Mock()]
        expected = True
        actual = CrunchCube({}).has_multiple_response
        assert actual == expected

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

    @patch('cr.cube.crunch_cube.CrunchCube._has_means', False)
    def test_missing(self):
        missing = Mock()
        fake_cube = {'result': {'missing': missing}}
        expected = missing
        actual = CrunchCube(fake_cube).missing
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube._has_means', True)
    def test_missing_with_means(self):
        missing = Mock()
        fake_cube = {'result': {'measures': {'mean': {'n_missing': missing}}}}
        expected = missing
        actual = CrunchCube(fake_cube).missing
        self.assertEqual(actual, expected)

    def test_has_means(self):
        has_means = Mock()
        with patch('cr.cube.crunch_cube.CrunchCube._has_means', has_means):
            expected = has_means
            actual = CrunchCube({}).has_means
            self.assertEqual(actual, expected)

    def test_test_filter_annotation(self):
        mock_cube = {'filter_names': Mock()}
        expected = mock_cube['filter_names']
        actual = CrunchCube(mock_cube).filter_annotation
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_y_offset_no_dimensions(self, mock_dims):
        dims = []
        mock_dims.__get__ = Mock(return_value=dims)
        expected = 4
        actual = CrunchCube({}).y_offset()
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.as_array')
    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_y_offset_two_dimensions(self, mock_dims, mock_as_array):
        dims = [Mock(), Mock()]
        mock_dims.__get__ = Mock(return_value=dims)
        mock_as_array.return_value = np.arange(12).reshape(3, 4)

        # Expected is the first dim length + offset of 4
        expected = 3 + 4
        actual = CrunchCube({}).y_offset()
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.as_array')
    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_y_offset_three_dimensions(self, mock_dims, mock_as_array):
        dims = [Mock(), Mock(), Mock()]
        mock_dims.__get__ = Mock(return_value=dims)
        mock_as_array.return_value = np.arange(24).reshape(3, 4, 2)

        # Expected is the first dim len * (sedond dim len + offset of 4)
        # The first dim is used as a slice, while the second is the number
        # of rows of each subtable. Offset is space between tables
        expected = 3 * (4 + 4)
        actual = CrunchCube({}).y_offset()
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.as_array')
    @patch('cr.cube.crunch_cube.CrunchCube.dimensions')
    def test_y_offset_ca_as_zero_index_cube(self, mock_dims, mock_as_array):
        dims = [Mock(), Mock()]
        dims[0].type = 'categorical_array'

        # Second dim has total length of 3
        dims[1].elements.return_value = [Mock(), Mock(), Mock()]
        mock_dims.__get__ = Mock(return_value=dims)
        mock_as_array.return_value = np.arange(12).reshape(3, 4)

        # Expected is the first dim len * (sedond dim len + offset of 4)
        # The first dim is used as a slice, while the second is the number
        # of rows of each subtable. Offset is space between tables
        expected = 3 * (3 + 4)
        # 'expand' argument expands the CA dim as slices
        actual = CrunchCube({}).y_offset(expand=True)
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
