'''Unit tests for the CrunchCube class.'''
from unittest import TestCase
from mock import Mock
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube


#pylint: disable=invalid-name
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

    @patch('cr.cube.crunch_cube.CrunchCube.as_array')
    @patch('cr.cube.crunch_cube.CrunchCube._is_double_multiple_response')
    @patch('cr.cube.crunch_cube.CrunchCube.margin')
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
        fake_adjusted = Mock()
        # Make the call
        cube.proportions(
            axis=fake_axis,
            weighted=fake_weighted,
            adjusted=fake_adjusted,
            include_transforms_for_dims=fake_dims
        )
        # Assert parameter propagation
        mock_margin.assert_called_once_with(
            axis=fake_axis,
            weighted=fake_weighted,
            adjusted=fake_adjusted,
            include_transforms_for_dims=fake_dims,
        )
