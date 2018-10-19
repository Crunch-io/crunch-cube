'''Unit tests for the CrunchCube class.'''

from unittest import TestCase
from mock import Mock
from mock import patch
import numpy as np

from cr.cube.crunch_cube import CrunchCube


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

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [])
    @patch('cr.cube.crunch_cube.CrunchCube.mr_selections_indices')
    def test_does_not_have_multiple_response(self, mock_mr_indices):
        mock_mr_indices.return_value = []
        expected = False
        actual = CrunchCube({}).has_mr
        self.assertEqual(actual, expected)

    @patch('cr.cube.crunch_cube.CrunchCube.mr_dim_ind', 0)
    def test_has_multiple_response(self):
        expected = True
        actual = CrunchCube({}).has_mr
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

    def test_has_means(self):
        has_means = Mock()
        with patch('cr.cube.crunch_cube.CrunchCube.has_means', has_means):
            expected = has_means
            actual = CrunchCube({}).has_means
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

    def test_insertions_with_empty_indices(self):
        cc = CrunchCube({})

        class DummyDimension:
            @property
            def hs_indices(self):
                return [{'anchor_ind': 0, 'inds': []}]

        result = Mock()
        dimension_index = 0
        dimension = DummyDimension()

        insertions = cc._insertions(result, dimension, dimension_index)
        assert insertions == [], insertions

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

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_cat_x_cat(self):
        '''Test axes for CAT x CAT.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction
        expected = (0,)
        actual = adjust(0)
        assert actual == expected

        # Test row direction
        expected = (1,)
        actual = adjust(1)
        assert actual == expected

        # Test table direction both None and (0, 1)
        expected = (0, 1)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((0, 1))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_cat_x_cat_x_cat(self):
        '''Test axes for CAT x CAT.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction
        expected = (0,)
        actual = adjust(0)
        assert actual == expected

        # Test col direction (3D => col == 1)
        expected = (1,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction (3D => row == 2)
        expected = (2,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction both None and (1, 2)
        expected = (1, 2)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((1, 2))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response'),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_univariate_mr(self):
        '''Test axes for univariate MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test table direction
        expected = (1,)
        actual = adjust(None)
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response'),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_cat_x_mr(self):
        '''Test axes for CAT x MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction
        expected = (0,)
        actual = adjust(0)
        assert actual == expected

        # Test row direction
        expected = (2,)
        actual = adjust(1)
        assert actual == expected

        # Test table direction
        expected = (0, 2)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((0, 1))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_mr_x_cat_x_cat(self):
        '''Test axes for MR x CAT x CAT.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test row direction
        expected = (2,)
        actual = adjust(1)
        assert actual == expected

        # Test table direction
        expected = (1, 2)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((0, 1))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_mr_x_mr(self):
        '''Test axes for MR x MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test row direction
        expected = (3,)
        actual = adjust(1)
        assert actual == expected

        # Test table direction
        expected = (1, 3)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((0, 1))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_cat_mr_x_mr(self):
        '''Test axes for CAT x MR x MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction (rarely used)
        expected = (0,)
        actual = adjust(0)
        assert actual == expected

        # Test col direction
        expected = (2,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction
        expected = (4,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction
        expected = (2, 4)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((1, 2))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_mr_x_cat_x_mr(self):
        '''Test axes for MR x CAT x MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction (rarely used)
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test col direction
        expected = (2,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction
        expected = (4,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction
        expected = (2, 4)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((1, 2))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_mr_x_mr_cat(self):
        '''Test axes for MR x MR x CAT.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction (rarely used)
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test col direction
        expected = (3,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction
        expected = (4,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction
        expected = (3, 4)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((1, 2))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_simple_ca(self):
        '''Test axes for simple CA.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col dimension (items - not allowed)
        with self.assertRaises(ValueError):
            adjust(0)

        # Test row direction (the only allowed direction, across categories)
        expected = (1,)
        actual = adjust(1)
        assert actual == expected

        # Test table direction - not allowed since cube contains CA items
        with self.assertRaises(ValueError):
            actual = adjust(None)
        with self.assertRaises(ValueError):
            adjust((0, 1))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_ca_x_cat(self):
        '''Test axes for CA x CAT.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th dimension (items - not allowed)
        with self.assertRaises(ValueError):
            adjust(0)

        # Test col direction
        expected = (1,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction
        expected = (2,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction - in this case table directions IS allowed,
        # since the items dimension is TABs, and will never be summed across
        expected = (1, 2)
        actual = adjust(None)
        assert actual == expected
        actual = adjust((1, 2))
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_cat_x_ca(self):
        '''Test axes for CAT x CA.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test col direction (not allowed across subvars)
        with self.assertRaises(ValueError):
            adjust(1)

        # Test row direction
        expected = (2,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction - not allowed - cube has CA items
        with self.assertRaises(ValueError):
            adjust(None)
        with self.assertRaises(ValueError):
            adjust((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_adjust_axis_ca_x_mr(self):
        '''Test axes for CAT x MR.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction (items - not allowed)
        with self.assertRaises(ValueError):
            adjust(0)

        # Test col direction
        expected = (1,)
        actual = adjust(1)
        assert actual == expected

        # Test row direction
        expected = (3,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction
        # In this case, table directions IS allowed, since it will never be
        # summed across (since it's TABs dimension)
        expected = (1, 3)
        actual = adjust(None)
        assert expected == actual
        actual = adjust((1, 2))
        assert expected == actual

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_adjust_axis_mr_x_cat_x_ca(self):
        '''Test axes for MR x CAT x CA.'''
        cc = CrunchCube({})
        adjust = cc._adjust_axis

        # Test 0th direction (rarely used)
        expected = (1,)
        actual = adjust(0)
        assert actual == expected

        # Test col direction (items - not allowed)
        with self.assertRaises(ValueError):
            adjust(1)

        # Test row direction
        expected = (3,)
        actual = adjust(2)
        assert actual == expected

        # Test table direction (doesn't need MR, since it's tabs)
        # Also: Not allowed for CA, since it has items dimension
        with self.assertRaises(ValueError):
            adjust(None)
        with self.assertRaises(ValueError):
            # If user wants to do the "table" direction by directly providing
            # both axes, he needs to know what he's doing. Otherwise, throw
            # and error, since adding across items (subvars) is not allowed.
            adjust((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_simple_ca(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0) is False
        assert is_allowed(1)
        assert is_allowed(None) is False

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_ca_x_cat(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0) is False
        assert is_allowed(1)
        assert is_allowed(2)
        assert is_allowed(None)
        assert is_allowed((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_cat_x_ca(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1) is False
        assert is_allowed(2)
        assert is_allowed(None) is False
        assert is_allowed((1, 2)) is False

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_mr_x_ca(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1) is False
        assert is_allowed(2)
        assert is_allowed(None) is False
        assert is_allowed((1, 2)) is False

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_axis_allowed_ca_x_mr(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0) is False
        assert is_allowed(1)
        assert is_allowed(2)
        assert is_allowed(None)
        assert is_allowed((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_univ_cat(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(None)

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_cat_x_cat(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1)
        assert is_allowed(None)
        assert is_allowed((0, 1))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_axis_allowed_cat_x_mr_x_mr(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1)
        assert is_allowed(2)
        assert is_allowed(None)
        assert is_allowed((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical', is_selections=False),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_axis_allowed_mr_x_cat_x_mr(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1)
        assert is_allowed(2)
        assert is_allowed(None)
        assert is_allowed((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='multiple_response', is_selections=False),
        Mock(type='categorical', is_selections=True),
        Mock(type='categorical', is_selections=False),
    ])
    def test_axis_allowed_mr_x_mr_x_cat(self):
        cc = CrunchCube({})
        is_allowed = cc._is_axis_allowed

        assert is_allowed(0)
        assert is_allowed(1)
        assert is_allowed(2)
        assert is_allowed(None)
        assert is_allowed((1, 2))

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical_array', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_ca_dim_ind_is_zero(self):
        cc = CrunchCube({})
        actual = cc.ca_dim_ind
        expected = 0
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical_array', is_selections=True),
    ])
    def test_ca_dim_ind_is_one(self):
        cc = CrunchCube({})
        actual = cc.ca_dim_ind
        expected = 1
        assert actual == expected

    @patch('cr.cube.crunch_cube.CrunchCube.all_dimensions', [
        Mock(type='categorical', is_selections=False),
        Mock(type='categorical', is_selections=True),
    ])
    def test_ca_dim_ind_is_none(self):
        cc = CrunchCube({})
        actual = cc.ca_dim_ind
        expected = None
        assert actual == expected

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
