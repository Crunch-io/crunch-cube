'''Unit tests for the CrunchCube class.'''
from unittest import TestCase
from mock import Mock
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube


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
