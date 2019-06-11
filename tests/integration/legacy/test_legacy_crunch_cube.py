from unittest import TestCase
import numpy as np
import pytest

from cr.cube.crunch_cube import (
    CrunchCube,
    _MeanMeasure,
    _Measures,
    _UnweightedCountMeasure,
    _WeightedCountMeasure,
)
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.measures.index import Index
from cr.cube.util import compress_pruned

from ...fixtures import CR  # ---mnemonic: CR = 'cube-response'---
from . import assert_scale_means_equal


class DescribeIntegratedCrunchCube(object):
    def it_provides_a_console_friendly_repr_for_a_cube(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        repr_ = repr(cube)
        assert repr_ == "CrunchCube(name='v4', dim_types='CAT x CAT')"

    def it_provides_access_to_its_dimensions(self, dimensions_fixture):
        cube_response, expected_dimension_types = dimensions_fixture
        cube = CrunchCube(cube_response)

        dimension_types = tuple(d.dimension_type for d in cube.dimensions)

        assert dimension_types == expected_dimension_types

    def it_knows_if_it_is_a_single_ca_cube(self):
        cube = CrunchCube(CR.SIMPLE_CAT_ARRAY)
        is_univariate_ca = cube.is_univariate_ca
        assert is_univariate_ca is True

    def it_knows_the_main_axis_of_a_univariate_ca_cube(self):
        cube = CrunchCube(CR.SIMPLE_CAT_ARRAY)  # ---CA_SUBVAR x CA_CAT---
        univariate_ca_main_axis = cube.univariate_ca_main_axis
        assert univariate_ca_main_axis == 1

    def it_provides_array_for_single_valid_cat_CAT_X_MR(self):
        # --The CAT dim of this cube has only 1 valid category (+1 missing).
        # --This is an edge case because that CAT dimension gets squashed.
        cube = CrunchCube(CR.CAT_X_MR_SENTRY)
        arr = cube.as_array(prune=True)
        np.testing.assert_array_equal(arr, np.array([[0, 0, 0]]))

    def it_provides_pruned_array_for_CA_CAT_x_CA_SUBVAR(self):
        cube = CrunchCube(CR.CA_CAT_X_CA_SUBVAR)
        arr = cube.proportions(axis=0, prune=True)
        np.testing.assert_array_equal(
            compress_pruned(arr),
            np.array(
                [
                    [0.19012797074954296, 0.10494203782794387],
                    [0.2528945764777575, 0.2190359975594875],
                    [0.16514320536258378, 0.1720561317876754],
                    [0.29859841560024375, 0.4069554606467358],
                    [0.09323583180987204, 0.09701037217815742],
                ]
            ),
        )

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            (CR.CA_SUBVAR_HS_X_MR_X_CA_CAT, (DT.CA_SUBVAR, DT.MR, DT.CA_CAT)),
            (CR.CAT_X_LOGICAL, (DT.CAT, DT.LOGICAL)),
            (CR.LOGICAL_UNIVARIATE, (DT.LOGICAL,)),
        ]
    )
    def dimensions_fixture(self, request):
        cube_response, expected_dimension_types = request.param
        return cube_response, expected_dimension_types


class DescribeIntegrated_Measures(object):
    def it_knows_when_its_measures_are_weighted(self, is_weighted_fixture):
        cube_dict, expected_value = is_weighted_fixture
        measures = _Measures(cube_dict, None)

        is_weighted = measures.is_weighted

        assert is_weighted == expected_value

    def it_provides_access_to_the_mean_measure(self):
        cube_dict = CR.CAT_X_CAT_MEAN_WGTD
        measures = _Measures(cube_dict, None)

        means = measures.means

        assert type(means).__name__ == "_MeanMeasure"

    def but_only_when_the_cube_response_contains_means(self):
        cube_dict = CR.CAT_X_CAT
        measures = _Measures(cube_dict, None)

        means = measures.means

        assert means is None

    def it_provides_the_means_missing_count_when_means_are_available(self):
        measures = _Measures(CR.CAT_X_CAT_MEAN_WGTD, None)
        missing_count = measures.missing_count
        assert missing_count == 3

    def but_provides_the_general_missing_count_otherwise(self):
        measures = _Measures(CR.CAT_X_CAT, None)
        missing_count = measures.missing_count
        assert missing_count == 5

    def it_knows_the_population_fraction(self, pop_frac_fixture):
        cube_dict, expected_value = pop_frac_fixture
        measures = _Measures(cube_dict, None)

        population_fraction = measures.population_fraction

        assert population_fraction == expected_value

    def it_provides_access_to_the_unweighted_count_measure(self):
        measures = _Measures(None, None)

        unweighted_counts = measures.unweighted_counts

        assert type(unweighted_counts).__name__ == "_UnweightedCountMeasure"

    def it_knows_the_unweighted_n(self):
        measures = _Measures(CR.CAT_X_CAT, None)
        unweighted_n = measures.unweighted_n
        assert unweighted_n == 20

    def it_provides_access_to_the_weighted_count_measure(self, wgtd_counts_fixture):
        cube_dict, expected_type_name = wgtd_counts_fixture
        measures = _Measures(cube_dict, None)

        weighted_counts = measures.weighted_counts

        assert type(weighted_counts).__name__ == expected_type_name

    def it_knows_the_weighted_n(self, wgtd_n_fixture):
        cube_dict, expected_value = wgtd_n_fixture
        measures = _Measures(cube_dict, None)

        weighted_n = measures.weighted_n

        assert round(weighted_n, 3) == expected_value

    # fixtures -------------------------------------------------------

    @pytest.fixture(
        params=[
            # ---has {'query': {'weight': url}}---
            (CR.ADMIT_X_GENDER_WEIGHTED, True),
            # ---has {'weight_var': weight_name_str}---
            (CR.CAT_X_CAT_X_CAT_WGTD, True),
            # ---unweighted_counts == measure_count_data---
            (CR.ADMIT_X_DEPT_UNWEIGHTED, False),
        ]
    )
    def is_weighted_fixture(self, request):
        cube_response, expected_value = request.param
        cube_dict = cube_response.get("value", cube_response)
        return cube_dict, expected_value

    @pytest.fixture(
        params=[
            # ---filtered case---
            (CR.CAT_X_CAT_FILT, 0.254),
            # ---unfiltered case---
            (CR.CAT_X_CAT, 1.0),
        ]
    )
    def pop_frac_fixture(self, request):
        cube_dict, expected_value = request.param
        return cube_dict, expected_value

    @pytest.fixture(
        params=[
            # ---weighted case---
            (CR.CAT_X_CAT_WGTD, "_WeightedCountMeasure"),
            # ---unweighted case---
            (CR.CAT_X_CAT, "_UnweightedCountMeasure"),
        ]
    )
    def wgtd_counts_fixture(self, request):
        cube_dict, expected_type_name = request.param
        return cube_dict, expected_type_name

    @pytest.fixture(
        params=[
            # ---weighted case---
            (CR.CAT_X_CAT_WGTD, 999.557),
            # ---unweighted case---
            (CR.CAT_X_CAT, 20.0),
        ]
    )
    def wgtd_n_fixture(self, request):
        cube_dict, expected_type = request.param
        return cube_dict, expected_type


class DescribeIntegrated_MeanMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT_MEAN_WGTD
        cube = CrunchCube(cube_dict)
        measure = _MeanMeasure(cube_dict, cube._all_dimensions)

        raw_cube_array = measure.raw_cube_array

        np.testing.assert_array_almost_equal(
            raw_cube_array,
            [
                [52.78205128, 49.90697674, np.nan, np.nan, np.nan],
                [50.43654822, 48.20100503, np.nan, np.nan, np.nan],
                [51.56435644, 47.60283688, np.nan, np.nan, np.nan],
                [58.0, 29.0, np.nan, np.nan, np.nan],
                [37.53846154, 39.45238095, np.nan, np.nan, np.nan],
                [36.66666667, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
            ],
        )


class DescribeIntegrated_UnweightedCountMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT
        cube = CrunchCube(cube_dict)
        measure = _UnweightedCountMeasure(cube_dict, cube._all_dimensions)

        raw_cube_array = measure.raw_cube_array

        np.testing.assert_array_almost_equal(
            raw_cube_array, [[5, 3, 2, 0], [5, 2, 3, 0], [0, 0, 0, 0]]
        )


class DescribeIntegrated_WeightedCountMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT_WGTD
        cube = CrunchCube(cube_dict)
        measure = _WeightedCountMeasure(cube_dict, cube._all_dimensions)

        raw_cube_array = measure.raw_cube_array

        np.testing.assert_array_almost_equal(
            raw_cube_array,
            [
                [32.9, 87.6, 176.2, 117.5, 72.1, 13.4, 0.0, 0.0, 0.0],
                [38.8, 94.1, 199.0128, 102.9, 38.8305, 26.2135, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )


class TestCrunchCube(TestCase):
    def test_crunch_cube_loads_data(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        cube_dict = cube._cube_dict
        self.assertEqual(cube_dict, CR.CAT_X_CAT)

    def test_as_array_univariate_cat_exclude_missing(self):
        cube = CrunchCube(CR.UNIVARIATE_CATEGORICAL)
        expected = np.array([10, 5])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_numeric(self):
        cube = CrunchCube(CR.VOTER_REGISTRATION)
        expected = np.array([885, 105, 10])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_datetime(self):
        cube = CrunchCube(CR.SIMPLE_DATETIME)
        expected = np.array([1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_text(self):
        cube = CrunchCube(CR.SIMPLE_TEXT)
        expected = np.array([1, 1, 1, 1, 1, 1])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[5, 2], [5, 3]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_cat_unweighted(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[5, 2], [5, 3]])
        actual = cube._as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_univariate_cat_axis_none(self):
        cube = CrunchCube(CR.UNIVARIATE_CATEGORICAL)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_numeric(self):
        cube = CrunchCube(CR.VOTER_REGISTRATION)
        expected = np.array([1000])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_datetime(self):
        cube = CrunchCube(CR.SIMPLE_DATETIME)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_text(self):
        cube = CrunchCube(CR.SIMPLE_TEXT)
        expected = np.array([6])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_none(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([15])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_none(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array([4])
        actual = cube.margin()
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_0(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([10, 5])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_0(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array([1, 1, 1, 1])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_cat_axis_1(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([7, 8])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_datetime_axis_1(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array([1, 1, 1, 1, 0])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_univariate_cat_axis_none(self):
        cube = CrunchCube(CR.UNIVARIATE_CATEGORICAL)
        expected = np.array([0.6666667, 0.3333333])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_numeric(self):
        cube = CrunchCube(CR.VOTER_REGISTRATION)
        expected = np.array([0.885, 0.105, 0.010])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_datetime(self):
        cube = CrunchCube(CR.SIMPLE_DATETIME)
        expected = np.array([0.25, 0.25, 0.25, 0.25])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_text(self):
        cube = CrunchCube(CR.SIMPLE_TEXT)
        expected = np.array(
            [0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_cat_axis_none(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[0.3333333, 0.1333333], [0.3333333, 0.2000000]])
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_none(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array(
            [
                [0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.25],
                [0.0, 0.25, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_cat_axis_0(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[0.5, 0.4], [0.5, 0.6]])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_0(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_cat_axis_1(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[0.7142857, 0.2857143], [0.6250000, 0.3750000]])
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_proportions_cat_x_datetime_axis_1(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_univariate_cat_axis_none(self):
        cube = CrunchCube(CR.UNIVARIATE_CATEGORICAL)
        expected = np.array([66.6666667, 33.3333333])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_numeric(self):
        cube = CrunchCube(CR.VOTER_REGISTRATION)
        expected = np.array([88.5, 10.5, 1.0])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_datetime(self):
        cube = CrunchCube(CR.SIMPLE_DATETIME)
        expected = np.array([25.0, 25.0, 25.0, 25.0])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_text(self):
        cube = CrunchCube(CR.SIMPLE_TEXT)
        expected = np.array(
            [16.6666667, 16.6666667, 16.6666667, 16.6666667, 16.6666667, 16.6666667]
        )
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_none(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[33.3333333, 13.3333333], [33.3333333, 20.0]])
        actual = cube.percentages()
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_0(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[50, 40], [50, 60]])
        actual = cube.percentages(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_percentages_cat_x_cat_axis_1(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[71.4285714, 28.5714286], [62.50000, 37.50000]])
        actual = cube.percentages(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_population_counts_univariate_cat(self):
        cube = CrunchCube(CR.UNIVARIATE_CATEGORICAL)
        expected = np.array([6000.6666667, 3000.3333333])
        actual = cube.population_counts(9001)
        np.testing.assert_almost_equal(actual, expected)

    def test_population_counts_numeric(self):
        cube = CrunchCube(CR.VOTER_REGISTRATION)
        expected = np.array([7965.885, 945.105, 90.01])
        actual = cube.population_counts(9001)
        np.testing.assert_almost_equal(actual, expected)

    def test_population_counts_datetime(self):
        cube = CrunchCube(CR.SIMPLE_DATETIME)
        expected = np.array([2250.25, 2250.25, 2250.25, 2250.25])
        actual = cube.population_counts(9001)
        np.testing.assert_almost_equal(actual, expected)

    def test_population_counts_text(self):
        cube = CrunchCube(CR.SIMPLE_TEXT)
        expected = np.array(
            [
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
            ]
        )
        actual = cube.population_counts(9001)
        np.testing.assert_almost_equal(actual, expected)

    def test_population_counts_cat_x_cat(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = np.array([[3000.3333333, 1200.1333333], [3000.3333333, 1800.2]])
        actual = cube.population_counts(9001)
        np.testing.assert_almost_equal(actual, expected)

    def test_filtered_population_counts(self):
        cube = CrunchCube(CR.CAT_X_CAT_FILT)
        expected = np.array(
            [
                [300000.0, 1400000.0, 0.0, 0.0, 0.0, 0.0],
                [5900000.0, 13200000.0, 0.0, 0.0, 0.0, 0.0],
                [600000.0, 2900000.0, 0.0, 0.0, 0.0, 0.0],
                [100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0],
                [300000.0, 600000.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        actual = cube.population_counts(100000000)
        np.testing.assert_almost_equal(actual, expected)

    def test_labels_cat_x_cat_exclude_missing(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = [["B", "C"], ["C", "E"]]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_cat_x_datetime_exclude_missing(self):
        cube = CrunchCube(CR.CAT_X_DATETIME)
        expected = [
            ["red", "green", "blue", "4", "9"],
            [
                "1776-07-04T00:00:00",
                "1950-12-24T00:00:00",
                "2000-01-01T00:00:00",
                "2000-01-02T00:00:00",
            ],
        ]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_labels_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(CR.SIMPLE_CAT_ARRAY)
        expected = [["ca_subvar_1", "ca_subvar_2", "ca_subvar_3"], ["a", "b", "c", "d"]]
        actual = cube.labels()
        self.assertEqual(actual, expected)

    def test_as_array_simple_cat_array_exclude_missing(self):
        cube = CrunchCube(CR.SIMPLE_CAT_ARRAY)
        expected = np.array([[3, 3, 0, 0], [1, 3, 2, 0], [0, 2, 1, 3]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_cat_x_num_x_datetime(self):
        """Test 3D cube, slicing accross first (numerical) variable."""
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        expected = np.array(
            [
                [[1, 1], [0, 0], [0, 0], [0, 0]],
                [[2, 1], [1, 1], [0, 0], [0, 0]],
                [[0, 0], [2, 3], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [3, 2], [0, 0]],
                [[0, 0], [0, 0], [1, 1], [0, 1]],
            ]
        )
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_num_datetime(self):
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        expected = np.array(
            [
                [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.4, 0.2], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.4, 0.6], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.6, 0.4], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.33333333, 0.33333333], [0.0, 0.33333333]],
            ]
        )
        # Set axis to tuple (1, 2), since we want to do a total for each slice
        # of the 3D cube. This is consistent with how the np.sum works
        # (axis parameter), which is used internally in
        # 'proportions' calculation.
        actual = cube.proportions((1, 2))
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_num_x_datetime_margin_by_table(self):
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        # Expect total for each slice (of which there are 5) rather than
        # expecting the total for the entire cube.
        expected = np.array([2, 5, 5, 5, 3])
        actual = cube.margin(axis=None)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_0(self):
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        # Expect resulting margin to have an additional dimension
        # (tabs CAT, with only a single element)
        expected = np.array([[[3, 2], [3, 4], [4, 3], [0, 1]]])
        actual = cube.margin(axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_1(self):
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        expected = np.array([[1, 1], [3, 2], [2, 3], [3, 2], [1, 2]])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_cat_x_num_x_datetime_axis_2(self):
        cube = CrunchCube(CR.CAT_X_NUM_X_DATETIME)
        expected = np.array(
            [[2, 0, 0, 0], [3, 2, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 2, 1]]
        )
        actual = cube.margin(axis=2)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_unweighted_gender_x_ideology(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([[32, 85, 171, 114, 70, 13], [40, 97, 205, 106, 40, 27]])
        actual = cube.as_array(weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_as_array_weighted_gender_x_ideology(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [
                    32.98969072,
                    87.62886598,
                    176.28865979,
                    117.5257732,
                    72.16494845,
                    13.40206186,
                ],
                [
                    38.83495146,
                    94.17475728,
                    199.02912621,
                    102.91262136,
                    38.83495146,
                    26.21359223,
                ],
            ]
        )
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_margin_weighted_gender_x_ideology_axis_0(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                71.82464218,
                181.80362326,
                375.31778601,
                220.43839456,
                110.99989991,
                39.61565409,
            ]
        )
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_0(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([72, 182, 376, 220, 110, 40])
        actual = cube.margin(axis=0, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_unweighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([485, 515])
        actual = cube.margin(axis=1, weighted=False)
        np.testing.assert_array_equal(actual, expected)

    def test_margin_weighted_gender_x_ideology_axis_1(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array([500, 500])
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_calculate_standard_error_axis_0(self):
        """Calculate standard error across columns."""
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [
                    -0.715899626017458,
                    -0.536708837208206,
                    -1.485149675785275,
                    1.114743776586886,
                    3.355236023985286,
                    -2.077040949965248,
                ],
                [
                    0.715899626017462,
                    0.536708837208211,
                    1.485149675785279,
                    -1.114743776586884,
                    -3.355236023985284,
                    2.07704094996525,
                ],
            ]
        )
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals(self):
        cube = CrunchCube(CR.CAT_X_CAT_GERMAN_WEIGHTED)
        expected = np.array(
            [
                [
                    0.1427612835278633,
                    0.0002121209149277,
                    0.0001314694879104,
                    0.1241771485531613,
                    0.0097454103240531,
                    0.0000000000000699,
                    0.8033849539190183,
                    0.4271118723152929,
                ],
                [
                    0.1427612835278667,
                    0.0002121209149277,
                    0.0001314694879104,
                    0.1241771485531642,
                    0.0097454103240533,
                    0.0000000000000699,
                    0.8033849539190183,
                    0.4271118723152956,
                ],
            ]
        )
        # Test without pruning
        actual = cube.pvals()
        np.testing.assert_almost_equal(actual, expected)

        # Test with pruning
        actual = cube.pvals(prune=True)
        np.testing.assert_almost_equal(actual, expected)

        # Test with pruning and H&S
        actual = cube.pvals(prune=True, hs_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_pvals_stats(self):
        cube = CrunchCube(CR.STATS_TEST)
        expected = np.array(
            [
                [
                    0.0436818197570077,
                    0.0002697141695955,
                    0.0913940671748992,
                    0.6836420776424197,
                    0.4681291494279529,
                    0.0000013632752629,
                ],
                [
                    0.0436818197570077,
                    0.0002697141695955,
                    0.0913940671748992,
                    0.6836420776424197,
                    0.4681291494279529,
                    0.0000013632752629,
                ],
            ]
        )
        # Test without pruning
        actual = cube.pvals()
        np.testing.assert_almost_equal(actual, expected)

        # Test with pruning
        actual = cube.pvals(prune=True)
        np.testing.assert_almost_equal(actual, expected)

        # Test with pruning and H&S
        actual = cube.pvals(prune=True, hs_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_mean_age_for_blame_x_gender(self):
        cube = CrunchCube(CR.ECON_MEAN_AGE_BLAME_X_GENDER)
        expected = np.array(
            [
                [52.78205128205122, 49.9069767441861],
                [50.43654822335009, 48.20100502512572],
                [51.5643564356436, 47.602836879432715],
                [58, 29],
                [37.53846153846155, 39.45238095238095],
            ]
        )
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_mean_no_dims(self):
        cube = CrunchCube(CR.ECON_MEAN_NO_DIMS)
        expected = np.array([49.095])
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_dept_unweighted_rows(self):
        """see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/zvalues-spec.js#L42
        """
        cube = CrunchCube(CR.ADMIT_X_DEPT_UNWEIGHTED)
        expected = np.array(
            [
                [
                    18.04029230689576,
                    13.018394979149804,
                    -2.576739836814013,
                    -3.059526328377408,
                    -7.230244530709987,
                    -19.321410263144653,
                ],
                [
                    -18.040292306895765,
                    -13.018394979149804,
                    2.576739836814013,
                    3.059526328377408,
                    7.230244530709987,
                    19.321410263144653,
                ],
            ]
        )
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_z_scores_admit_by_gender_weighted_rows(self):
        """ see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/zvalues-spec.js#L67
        """
        cube = CrunchCube(CR.ADMIT_X_GENDER_WEIGHTED)
        expected = np.array(
            [
                [9.42561984520692, -9.425619845206922],
                [-9.425619845206922, 9.42561984520692],
            ]
        )
        actual = cube.zscore()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_dim_names(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = ["Statements agreed with about Climate", "Gender"]
        actual = [dim.name for dim in cube.dimensions]
        self.assertEqual(actual, expected)

    def test_selected_crosstab_as_array(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [9928.20954289002, 11524.821237084192],
                [9588.843313998908, 9801.254016136965],
                [11697.435357575358, 13095.670425525452],
                [9782.8995547749, 10531.918128023966],
                [4417.596222134318, 3448.380316269752],
                [6179.175512581436, 6490.427474934746],
            ]
        )
        actual = cube.as_array()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_by_rows(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                21453.03077997421,
                19390.097330135875,
                24793.105783100807,
                20314.817682798865,
                7865.976538404069,
                12669.602987516182,
            ]
        )
        actual = cube.margin(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_by_cols(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [14566.261567907562, 15607.301233922663],
                [14456.513325488017, 15450.609903833058],
                [14415.136475733132, 15405.898678070093],
                [11485.661204663904, 11912.588886491172],
                [11664.69933815247, 12110.196347286023],
                [11547.413553551738, 11961.575582997419],
            ]
        )
        actual = cube.margin(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_margin_total(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                30173.5628018302,
                29907.1232293211,
                29821.0351538032,
                23398.2500911551,
                23774.8956854385,
                23508.9891365492,
            ]
        )
        actual = cube.margin()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_by_rows(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [0.4627882020361299, 0.5372117979638701],
                [0.4945227014975337, 0.5054772985024663],
                [0.47180193800279874, 0.5281980619972013],
                [0.481564723224583, 0.5184352767754171],
                [0.5616081106479636, 0.4383918893520365],
                [0.48771658580541166, 0.5122834141945883],
            ]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_by_cols(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [0.6815894041587091, 0.7384249886863752],
                [0.6632887957217867, 0.6343603312193796],
                [0.8114689290154947, 0.8500426167391849],
                [0.8517489224566737, 0.8840998567462627],
                [0.3787149667617584, 0.28475015741941767],
                [0.535113381358101, 0.5426064007955989],
            ]
        )
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_proportions_by_cell(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = np.array(
            [
                [0.329036700375595, 0.381950958618156],
                [0.320620717695708, 0.327723062528721],
                [0.392254504152701, 0.439142047148397],
                [0.418103897371069, 0.450115632023491],
                [0.185809278853744, 0.14504292098248],
                [0.262843097025161, 0.27608279697761],
            ]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_selected_crosstab_is_double_mr(self):
        cube = CrunchCube(CR.SELECTED_CROSSTAB_4)
        expected = False
        actual = cube.is_double_mr
        self.assertEqual(actual, expected)

    def test_pets_x_pets_is_double_mr(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        is_double_mr = cube.is_double_mr
        self.assertEqual(is_double_mr, True)

    def test_pets_x_pets_as_array(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        expected = np.array([[40, 14, 18], [14, 34, 16], [18, 16, 38]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_x_pets_proportions_by_cell(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        expected = np.array(
            [
                [0.5, 0.2, 0.2571429],
                [0.2, 0.4303797, 0.2285714],
                [0.2571429, 0.2285714, 0.5428571],
            ]
        )
        actual = cube.proportions()
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_proportions_by_col(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        expected = np.array(
            [
                [1.0, 0.4827586, 0.4736842],
                [0.4117647, 1.0, 0.4210526],
                [0.5294118, 0.5517241, 1.0],
            ]
        )
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_proportions_by_row(self):
        cube = CrunchCube(CR.PETS_X_PETS)
        expected = np.array(
            [
                [1.0, 0.4117647, 0.5294118],
                [0.4827586, 1.0, 0.5517241],
                [0.4736842, 0.4210526, 1.0],
            ]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_as_array(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = np.array([[12, 28], [12, 22], [12, 26]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_x_fruit_margin_row(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = np.array([40, 34, 38])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_as_array(self):
        cube = CrunchCube(CR.PETS_ARRAY)
        expected = np.array([[45, 34], [40, 40], [32, 38]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_proportions(self):
        cube = CrunchCube(CR.PETS_ARRAY)
        expected = np.array(
            [[0.5696203, 0.4303797], [0.5000000, 0.500000], [0.4571429, 0.5428571]]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_proportions_bad_directions(self):
        cube = CrunchCube(CR.PETS_ARRAY)
        with self.assertRaises(ValueError):
            cube.proportions(axis=0)

        # This is not bad direction, because the 'None' always
        # figures out what it needs to do
        # with self.assertRaises(ValueError):
        #     cube.proportions(axis=None)

    def test_pets_array_margin_total(self):
        cube = CrunchCube(CR.PETS_ARRAY)
        with self.assertRaises(ValueError):
            cube.margin()

    def test_pets_array_margin_by_row(self):
        cube = CrunchCube(CR.PETS_ARRAY)
        expected = np.array([79, 80, 70])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_pets_array_margin_by_col_not_allowed_across_items(self):
        """Colum direction is not allowed across items dimension."""
        cube = CrunchCube(CR.PETS_ARRAY)
        with self.assertRaises(ValueError):
            cube.margin(axis=0)

    def test_count_unweighted(self):
        cube = CrunchCube(CR.ADMIT_X_GENDER_WEIGHTED)
        expected = 4526
        actual = cube.count(weighted=False)
        self.assertEqual(actual, expected)

    def test_count_weighted(self):
        cube = CrunchCube(CR.ADMIT_X_GENDER_WEIGHTED)
        expected = 4451.955438803248
        actual = cube.count(weighted=True)
        self.assertEqual(actual, expected)

    def test_econ_x_ideology_index_by_col(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [0.91861761, 0.96399471, 0.9394101, 1.06629132, 1.30027051, 0.67660435],
                [1.08138239, 1.03600529, 1.0605899, 0.93370868, 0.69972949, 1.32339565],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_econ_x_ideology_index_by_row(self):
        cube = CrunchCube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED)
        expected = np.array(
            [
                [0.91861761, 0.96399471, 0.9394101, 1.06629132, 1.30027051, 0.67660435],
                [1.08138239, 1.03600529, 1.0605899, 0.93370868, 0.69972949, 1.32339565],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_cell(self):
        cube = CrunchCube(CR.FRUIT_X_PETS)
        expected = np.array(
            [[0.15, 0.15189873, 0.17142857], [0.35, 0.27848101, 0.37142857]]
        )
        actual = cube.proportions(axis=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_row(self):
        cube = CrunchCube(CR.FRUIT_X_PETS)
        expected = np.array(
            [[0.4285714, 0.48, 0.5217391], [0.5384615, 0.4074074, 0.5531915]]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_fruit_x_pets_proportions_by_col(self):
        cube = CrunchCube(CR.FRUIT_X_PETS)
        expected = np.array([[0.3, 0.3529412, 0.3157895], [0.7, 0.6470588, 0.6842105]])
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_cell(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = np.array(
            [[0.15, 0.35], [0.15189873, 0.27848101], [0.17142857, 0.37142857]]
        )
        actual = cube.proportions(axis=None)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_col(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = np.array(
            [[0.4285714, 0.5384615], [0.48, 0.4074074], [0.5217391, 0.5531915]]
        )
        actual = cube.proportions(axis=0)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_fruit_proportions_by_row(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = np.array(
            [[0.3, 0.7], [0.3529412, 0.6470588], [0.3157895, 0.6842105]]
        )
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_cat_x_cat_array_proportions_by_row(self):
        """Get the proportions for each slice of the 3D cube.

        The axis is equal to 2, since this is the dimensions across which
        we have to calculate the margin.
        """
        cube = CrunchCube(CR.FRUIT_X_PETS_ARRAY)
        expected = [
            [[0.52, 0.48], [0.57142857, 0.42857143], [0.47826087, 0.52173913]],
            [
                [0.59259259, 0.40740741],
                [0.46153846, 0.53846154],
                [0.44680851, 0.55319149],
            ],
        ]
        actual = cube.proportions(axis=2)
        np.testing.assert_almost_equal(actual, expected)

    def test_identity_x_period_axis_out_of_bounds(self):
        cube = CrunchCube(CR.NUM_X_NUM_EMPTY)
        # There are margins that have 0 value in this cube. In whaam, they're
        # pruned, so they're not shown. CrunchCube is not responsible for
        # pruning (cr.exporter is).
        expected = np.array([94, 0, 248, 210, 102, 0, 0, 0, 286, 60])
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_with_single_cat(self):
        cube = CrunchCube(CR.CA_SINGLE_CAT)
        # The last 0 of the expectation is not visible in whaam because of
        # pruning, which is not the responsibility of cr.cube.
        expected = np.array([79, 80, 70, 0])
        actual = cube.margin(axis=1, weighted=False)
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_by_col(self):
        cube = CrunchCube(CR.PETS_ARRAY_X_PETS)
        expected = np.array(
            [[0.59097127, 0.0, 0.55956679], [0.40902873, 1.0, 0.44043321]]
        )
        # Since cube is 3D, col dim is 1 (instead of 0)
        actual = cube.proportions(axis=1)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_row(self):
        cube = CrunchCube(CR.PETS_ARRAY_X_PETS)
        expected = np.array(
            [[0.44836533, 0.0, 0.48261546], [0.39084967, 1.0, 0.47843137]]
        )
        # Since cube is 3D, row dim is 2 (instead of 0)
        actual = cube.proportions(axis=2)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_array_x_pets_cell(self):
        cube = CrunchCube(CR.PETS_ARRAY_X_PETS)
        expected = np.array(
            [[0.24992768, 0.00000000, 0.26901938], [0.17298235, 0.44258027, 0.21174429]]
        )
        actual = cube.proportions(axis=None)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_pets_x_pets_array_margin_by_cell(self):
        cube = CrunchCube(CR.PETS_X_PETS_ARRAY)
        # TODO: Confirm with Jon and Mike
        # This margin is not something a user would expect, because it
        # performs summation across the items dimension of the CA.
        # expected = np.array([229, 229, 229])
        with self.assertRaises(ValueError):
            cube.margin()

    def test_pets_x_pets_array_percentages(self):
        """All directions need to return same percentages.

        The only direction that makes sense is across categories, and this is
        handled automatically by the cube.
        """
        cube = CrunchCube(CR.PETS_X_PETS_ARRAY)
        expected = np.array(
            [
                [0.58823529, 0.41176471],
                [0.00000000, 1.00000000],
                [0.47058824, 0.52941176],
            ]
        )
        # TODO: Remove this if not needed anymore...
        # TODO: Consult with jon and Mike. The new expectation is closer to what
        # whaam displays, but diverges from R.
        # expected = np.array([
        #     [0.55555556, 0.19444444],
        #     [0., 0.55555556],
        #     [0.44444444, 0.25],
        # ])

        # The direction 1 designates columns (for each slice). Since this is
        # the subvar dimension, it needs to be treated as invalid direction
        with self.assertRaises(ValueError):
            cube.proportions(axis=1)

        # The direction None designates "table" (total for each slice). Since
        # this would contain the subvar dimension, it needs to be treated as
        # invalid direction
        with self.assertRaises(ValueError):
            cube.proportions()[0]

        actual = cube.proportions(axis=2)[0]
        np.testing.assert_almost_equal(actual, expected)

    def test_profiles_percentages_add_up_to_100(self):
        cube = CrunchCube(CR.PROFILES_PERCENTS)
        props = cube.percentages(axis=1)
        actual_sum = np.sum(props, axis=1)
        expected_sum = np.ones(props.shape[0]) * 100
        np.testing.assert_almost_equal(actual_sum, expected_sum)

    def test_cat_x_cat_as_array_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [2, 2, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 2],
                [0, 2, 0, 0],
                [0, 2, 0, 1],
                [0, 1, 0, 0],
            ]
        )
        actual = cube.as_array(prune=False)
        expected = np.array([[2, 2, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 1, 0]])
        table = cube.as_array(prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_cat_x_cat_props_by_col_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [1.0, 0.25, np.nan, 0.25],
                [0.0, 0.0, np.nan, 0.0],
                [0.0, 0.125, np.nan, 0.5],
                [0.0, 0.25, np.nan, 0.0],
                [0.0, 0.25, np.nan, 0.25],
                [0.0, 0.125, np.nan, 0.0],
            ]
        )
        actual = cube.proportions(axis=0, prune=False)
        expected = np.array(
            [
                [1.0, 0.25, 0.25],
                [0.0, 0.125, 0.5],
                [0.0, 0.25, 0.0],
                [0.0, 0.25, 0.25],
                [0.0, 0.125, 0.0],
            ]
        )
        table = cube.proportions(axis=0, prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_array_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_cat_x_cat_props_by_row_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [0.4, 0.4, 0.0, 0.2],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.33333333, 0.0, 0.66666667],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.66666667, 0.0, 0.33333333],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        actual = cube.proportions(axis=1, prune=False)
        expected = np.array(
            [
                [0.4, 0.4, 0.2],
                [0.0, 0.33333333, 0.66666667],
                [0.0, 1.0, 0.0],
                [0.0, 0.66666667, 0.33333333],
                [0.0, 1.0, 0.0],
            ]
        )
        table = cube.proportions(axis=1, prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_almost_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_cat_x_cat_props_by_cell_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [0.14285714, 0.14285714, 0.0, 0.07142857],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.07142857, 0.0, 0.14285714],
                [0.0, 0.14285714, 0.0, 0.0],
                [0.0, 0.14285714, 0.0, 0.07142857],
                [0.0, 0.07142857, 0.0, 0.0],
            ]
        )
        actual = cube.proportions(axis=None, prune=False)
        expected = np.array(
            [
                [0.14285714, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.14285714],
                [0.0, 0.14285714, 0.0],
                [0.0, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.0],
            ]
        )
        table = cube.proportions(axis=None, prune=True)
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_almost_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_cat_x_cat_index_by_col_prune_cols(self):
        cube = CrunchCube(CR.CAT_X_CAT_WITH_EMPTY_COLS)
        expected = np.array(
            [
                [2.8, 0.7, np.nan, 0.7],
                [np.nan, np.nan, np.nan, np.nan],
                [0.0, 0.58333333, np.nan, 2.33333333],
                [0.0, 1.75, np.nan, 0.0],
                [0.0, 1.16666667, np.nan, 1.16666667],
                [0.0, 1.75, np.nan, 0.0],
            ]
        )
        actual = Index.data(cube, weighted=True, prune=False)
        # Assert index without pruning
        np.testing.assert_almost_equal(actual, expected)

        expected = np.array(
            [
                [2.8, 0.7, 0.7],
                [0.0, 0.58333333, 2.33333333],
                [0.0, 1.75, 0.0],
                [0.0, 1.16666667, 1.16666667],
                [0.0, 1.75, 0.0],
            ]
        )
        table = Index.data(cube, weighted=True, prune=True)
        # Assert index witih pruning
        actual = table[:, ~table.mask.all(axis=0)][~table.mask.all(axis=1), :]
        np.testing.assert_almost_equal(actual, expected)

        pruned_expected = [
            np.array([False, True, False, False, False, False]),
            np.array([False, False, True, False]),
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_prune_univariate_cat(self):
        cube = CrunchCube(CR.BINNED)
        expected = np.array([118504.40402204, 155261.2723631, 182923.95470245])
        actual = cube.as_array(prune=True)
        np.testing.assert_almost_equal(actual[~actual.mask], expected)

        pruned_expected = [
            np.array(
                [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ]
            )
        ]
        pruned = cube.prune_indices()
        self.assertEqual(len(pruned), len(pruned_expected))
        for i, actual in enumerate(pruned):
            np.testing.assert_array_equal(pruned[i], pruned_expected[i])

    def test_single_col_margin_not_iterable(self):
        cube = CrunchCube(CR.SINGLE_COL_MARGIN_NOT_ITERABLE)
        expected = (1,)
        actual = cube.margin(axis=0).shape
        self.assertEqual(actual, expected)

    def test_3d_percentages_by_col(self):
        # ---CAT x CAT x CAT---
        cube = CrunchCube(CR.GENDER_PARTY_RACE)
        expected = np.array(
            [
                [
                    [0.17647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.17647059, 0.05882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.23529412, 0.0, 0.0, 0.0, 0.0, 0.05882353, 0.0, 0.0],
                    [
                        0.11764706,
                        0.05882353,
                        0.0,
                        0.05882353,
                        0.0,
                        0.05882353,
                        0.0,
                        0.0,
                    ],
                ],
                [
                    [0.04761905, 0.0, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0],
                    [
                        0.14285714,
                        0.04761905,
                        0.0952381,
                        0.04761905,
                        0.0,
                        0.04761905,
                        0.0,
                        0.0,
                    ],
                    [0.23809524, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.19047619, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        # Set axis to tuple (1, 2), since we want to do a total for each slice
        # of the 3D cube. This is consistent with how the np.sum works
        # (axis parameter), which is used internally in
        # 'proportions' calculation.
        axis = (1, 2)
        actual = cube.proportions(axis=axis)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_dim_ind_for_cat_cube(self):
        cube = CrunchCube(CR.CAT_X_CAT)
        expected = None
        actual = cube.mr_dim_ind
        self.assertEqual(actual, expected)

    def test_total_unweighted_margin_when_has_means(self):
        """Tests that total margin is Unweighted N, when cube has means."""
        cube = CrunchCube(CR.CAT_MEAN_WGTD)
        expected = 17615
        actual = cube.margin(weighted=False)
        assert actual == expected

    def test_row_unweighted_margin_when_has_means(self):
        """Tests that total margin is Unweighted N, when cube has means."""
        cube = CrunchCube(CR.CAT_MEAN_WGTD)
        expected = np.array(
            [
                806,
                14,
                14,
                28,
                780,
                42,
                1114,
                28,
                24,
                746,
                2,
                12,
                6,
                2178,
                2026,
                571,
                136,
                16,
                14,
                1334,
                1950,
                26,
                128,
                4,
                28,
                3520,
                1082,
                36,
                56,
                556,
                38,
                146,
                114,
                28,
                12,
            ]
        )
        actual = np.ma.compressed(cube.margin(axis=1, weighted=False, prune=True))
        np.testing.assert_array_equal(actual, expected)
        # not testing cube.prune_indices() because the margin has 6367 cells

    def test_ca_with_single_cat_pruning(self):
        cube = CrunchCube(CR.CA_SINGLE_CAT)
        # The last 0 of the expectation is not visible in whaam because of
        # pruning, which is not the responsibility of cr.cube.
        expected = np.array([79, 80, 70])
        actual = np.ma.compressed(cube.as_array(weighted=False, prune=True))
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_single_cat_counts(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)

        expected = np.array([[13, 12], [16, 12], [11, 12]])
        actual = cube.as_array()
        np.testing.assert_array_equal(actual, expected)

        # Expectation for pruned indices (none are pruned)
        expecteds = [
            (np.array([False, False]), np.array([False])),
            (np.array([False, False]), np.array([False])),
            (np.array([False, False]), np.array([False])),
        ]
        actuals = cube.prune_indices()
        for expected, actual in zip(expecteds, actuals):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_ca_x_single_cat_props_by_col(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array(
            [[0.52, 0.48], [0.57142857, 0.42857143], [0.47826087, 0.52173913]]
        )
        # Col direction is 1 (and not 0) because of 3D cube.
        actual = cube.proportions(axis=1)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_x_single_cat_props_by_row(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        # Col direction is 2 (and not 1) because of 3D cube.
        actual = cube.proportions(axis=2)
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_x_single_cat_props_by_cell(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array(
            [[0.52, 0.48], [0.57142857, 0.42857143], [0.47826087, 0.52173913]]
        )
        # Col direction is (1, 2) because 3D cube (total per slice).
        actual = cube.proportions(axis=(1, 2))
        np.testing.assert_almost_equal(actual, expected)

    def test_ca_x_single_cat_col_margins(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array([25, 28, 23])
        # Axis equals to 1, because col direction in 3D cube is 1 (and not 0).
        # It operates on the 0th dimension of each slice (which is 1st
        # dimension of the cube).
        actual = cube.margin(axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_single_cat_row_margins(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array([[13, 12], [16, 12], [11, 12]])
        # Axis equals to 2, because col direction in 3D cube is 2 (and not 1).
        # It operates on the 1st dimension of each slice (which is 2nd
        # dimension of the cube).
        actual = cube.margin(axis=2)
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_single_cat_cell_margins(self):
        cube = CrunchCube(CR.CA_X_SINGLE_CAT)
        expected = np.array([25, 28, 23])
        # Axis equals to (1, 2), because the total is calculated for each slice.
        actual = cube.margin(axis=(1, 2))
        np.testing.assert_array_equal(actual, expected)

    def test_ca_subvar_x_cat_hs_counts_prune(self):
        cube = CrunchCube(CR.CA_SUBVAR_X_CAT_HS)
        expected = np.array([[3, 3, 0, 0, 6], [1, 3, 2, 0, 4], [0, 2, 1, 3, 2]])
        actual = cube.as_array(include_transforms_for_dims=[0, 1], prune=True)
        np.testing.assert_array_equal(actual, expected)

    def test_means_univariate_cat(self):
        cube = CrunchCube(CR.ECON_BLAME_WITH_HS)
        expected = [[np.array(2.1735205616850553)]]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_bivariate_cat(self):
        cube = CrunchCube(CR.ECON_BLAME_X_IDEOLOGY_ROW_HS)
        expected = [
            [
                np.array(
                    [2.19444444, 2.19230769, 2.26666667, 1.88990826, 1.76363636, 3.85]
                )
            ]
        ]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_x_mr(self):
        cube = CrunchCube(CR.FRUIT_X_PETS)
        expected = [[np.array([1.7, 1.6470588, 1.6842105]), None]]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_mr_x_cat(self):
        cube = CrunchCube(CR.PETS_X_FRUIT)
        expected = [[None, np.array([1.7, 1.6470588, 1.6842105])]]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_array_cat_dim_first(self):
        cube = CrunchCube(CR.PETS_ARRAY_CAT_FIRST)
        expected = [[None, np.array([1.44333002, 1.48049069, 1.57881177])]]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_array_subvar_dim_first(self):
        cube = CrunchCube(CR.PETS_ARRAY_SUBVAR_FIRST)
        expected = [[np.array([1.44333002, 1.48049069, 1.57881177]), None]]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_x_cat_arr_fruit_first(self):
        cube = CrunchCube(CR.FRUIT_X_PETS_ARRAY)
        expected = [
            [None, np.array([1.48, 1.42857143, 1.52173913])],
            [None, np.array([1.40740741, 1.53846154, 1.55319149])],
        ]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_x_cat_arr_subvars_first(self):
        cube = CrunchCube(CR.FRUIT_X_PETS_ARRAY_SUBVARS_FIRST)
        expected = [
            [np.array([1.71111111, 1.6, 1.65625]), None],
            [np.array([1.64705882, 1.7, 1.68421053]), None],
        ]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_cat_x_cat_arr_pets_first(self):
        cube = CrunchCube(CR.FRUIT_X_PETS_ARRAY_PETS_FIRST)
        expected = [
            [np.array([1.48, 1.40740741]), np.array([1.71111111, 1.64705882])],
            [np.array([1.42857143, 1.53846154]), np.array([1.6, 1.7])],
            [np.array([1.52173913, 1.55319149]), np.array([1.65625, 1.68421053])],
        ]
        actual = cube.scale_means()
        assert_scale_means_equal(actual, expected)

    def test_means_with_null_values(self):
        cube = CrunchCube(CR.SCALE_WITH_NULL_VALUES)
        scale_means = cube.scale_means()
        assert_scale_means_equal(
            scale_means, [[np.array([1.2060688, 1.0669344, 1.023199]), None]]
        )

    def test_values_services(self):
        cube = CrunchCube(CR.MR_X_CA_CAT_X_CA_SUBVAR)
        # Axis is 1, which is 'col' direction, since 3D cube.
        actual = cube.proportions(axis=1)[0]
        expected = np.array(
            [
                [
                    0.14285714,
                    0.10204082,
                    0.20512821,
                    0.16363636,
                    0.16438356,
                    0.1372549,
                    0.18181818,
                    0.2991453,
                    0.32,
                    0.44776119,
                ],  # noqa
                [
                    0.07142857,
                    0.23469388,
                    0.17948718,
                    0.14545455,
                    0.20547945,
                    0.09803922,
                    0.27272727,
                    0.11111111,
                    0.352,
                    0.23880597,
                ],  # noqa
                [
                    0.12857143,
                    0.19387755,
                    0.1025641,
                    0.16363636,
                    0.1369863,
                    0.15686275,
                    0.25,
                    0.17094017,
                    0.136,
                    0.14925373,
                ],  # noqa
                [
                    0.15714286,
                    0.15306122,
                    0.14102564,
                    0.05454545,
                    0.17808219,
                    0.09803922,
                    0.18181818,
                    0.20512821,
                    0.064,
                    0.05223881,
                ],  # noqa
                [
                    0.12857143,
                    0.12244898,
                    0.1025641,
                    0.05454545,
                    0.15068493,
                    0.07843137,
                    0.06060606,
                    0.1025641,
                    0.064,
                    0.05970149,
                ],  # noqa
                [
                    0.05714286,
                    0.09183673,
                    0.20512821,
                    0.09090909,
                    0.09589041,
                    0.11764706,
                    0.03030303,
                    0.02564103,
                    0.032,
                    0.01492537,
                ],  # noqa
                [
                    0.08571429,
                    0.04081633,
                    0.05128205,
                    0.07272727,
                    0.01369863,
                    0.11764706,
                    0.01515152,
                    0.05128205,
                    0.024,
                    0.02238806,
                ],  # noqa
                [
                    0.17142857,
                    0.04081633,
                    0.01282051,
                    0.03636364,
                    0.02739726,
                    0.01960784,
                    0.00757576,
                    0.00854701,
                    0.008,
                    0.00746269,
                ],  # noqa
                [
                    0.01428571,
                    0.02040816,
                    0.0,
                    0.14545455,
                    0.01369863,
                    0.11764706,
                    0.0,
                    0.0,
                    0.0,
                    0.00746269,
                ],  # noqa
                [
                    0.04285714,
                    0.0,
                    0.0,
                    0.07272727,
                    0.01369863,
                    0.05882353,
                    0.0,
                    0.02564103,
                    0.0,
                    0.0,
                ],  # noqa
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(actual, expected)
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_props_with_hs_by_cell(self):
        cube = CrunchCube(CR.LETTERS_X_PETS_HS)
        expected = np.array(
            [
                [
                    0.10769231,
                    0.16923077,
                    0.27692308,
                    0.26153846,
                    0.15384615,
                    0.15384615,
                ],  # noqa
                [
                    0.11111111,
                    0.20634921,
                    0.31746032,
                    0.19047619,
                    0.15873016,
                    0.15873016,
                ],  # noqa
                [
                    0.09090909,
                    0.22727273,
                    0.31818182,
                    0.24242424,
                    0.12121212,
                    0.12121212,
                ],  # noqa
                [
                    0.10447761,
                    0.14925373,
                    0.25373134,
                    0.13432836,
                    0.17910448,
                    0.17910448,
                ],  # noqa
                [
                    0.07462687,
                    0.11940299,
                    0.19402985,
                    0.23880597,
                    0.1641791,
                    0.1641791,
                ],  # noqa
            ]
        )
        actual = cube.proportions(include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_props_with_hs_by_row(self):
        cube = CrunchCube(CR.LETTERS_X_PETS_HS)
        expected = np.array(
            [
                [0.15555556, 0.24444444, 0.4, 0.37777778, 0.22222222, 0.22222222],
                [
                    0.16666667,
                    0.30952381,
                    0.47619048,
                    0.28571429,
                    0.23809524,
                    0.23809524,
                ],  # noqa
                [
                    0.13333333,
                    0.33333333,
                    0.46666667,
                    0.35555556,
                    0.17777778,
                    0.17777778,
                ],  # noqa
                [
                    0.18421053,
                    0.26315789,
                    0.44736842,
                    0.23684211,
                    0.31578947,
                    0.31578947,
                ],  # noqa
                [0.125, 0.2, 0.325, 0.4, 0.275, 0.275],
            ]
        )
        actual = cube.proportions(axis=1, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_mr_props_with_hs_by_col(self):
        cube = CrunchCube(CR.LETTERS_X_PETS_HS)
        expected = np.array(
            [
                [
                    0.53846154,
                    0.6875,
                    0.62068966,
                    0.94444444,
                    0.55555556,
                    0.55555556,
                ],  # noqa
                [
                    0.58333333,
                    0.68421053,
                    0.64516129,
                    0.66666667,
                    0.71428571,
                    0.71428571,
                ],  # noqa
                [0.5, 0.78947368, 0.67741935, 0.76190476, 0.57142857, 0.57142857],
                [0.53846154, 0.58823529, 0.56666667, 0.5, 0.63157895, 0.63157895],
                [
                    0.45454545,
                    0.47058824,
                    0.46428571,
                    0.76190476,
                    0.61111111,
                    0.61111111,
                ],  # noqa
            ]
        )
        actual = cube.proportions(axis=0, include_transforms_for_dims=[0, 1])
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_pruning_indices(self):
        """Test pruning indices for a simple XYZ cube."""
        cube = CrunchCube(CR.XYZ_SIMPLE_ALLTYPES)

        # Zeroth slice of the XYZ array:
        #
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  1  | 0  | False
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        # |  0 |  0  | 0  | True
        # +----+-----+----+
        #  True False True

        expected = (
            np.array([True, True, False, True, True]),
            np.array([True, False, True]),
        )
        actual = cube.prune_indices()[0]
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_mr_x_ca_rows_margin(self):
        cube = CrunchCube(CR.MR_X_CA_HS)
        actual = cube.margin(axis=2)
        expected = np.array([[3, 3, 3], [4, 4, 4], [0, 0, 0]])
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_mr_margin(self):
        cube = CrunchCube(CR.CA_X_MR_WEIGHTED_HS)
        expected = np.array([504, 215, 224, 76, 8, 439])
        actual = cube.margin(
            axis=1, weighted=False, include_transforms_for_dims=[0, 1, 2]
        )[0]
        np.testing.assert_array_equal(actual, expected)

    def test_ca_x_mr_margin_prune(self):
        # ---CA x MR---
        cube = CrunchCube(CR.CA_X_MR_WEIGHTED_HS)

        margin = cube.margin(
            axis=1, weighted=False, include_transforms_for_dims=[0, 1, 2], prune=True
        )[0]

        np.testing.assert_array_equal(margin, np.array([504, 215, 224, 76, 8, 439]))

    def test_mr_x_cat_x_mr_pruning(self):
        cube = CrunchCube(CR.MR_X_CAT_X_MR_PRUNE)
        expected = np.array(
            [
                [False, False, False, True],
                [False, False, False, True],
                [True, True, True, True],
                [False, False, False, True],
                [False, False, False, True],
                [False, False, False, True],
                [True, True, True, True],
                [True, True, True, True],
            ]
        )
        actual = cube.proportions(prune=True)[0].mask
        np.testing.assert_array_equal(actual, expected)

    def test_mr_x_mr_prune_indices(self):
        cube = CrunchCube(CR.HUFFPOST_ACTIONS_X_HOUSEHOLD)
        expected = [
            np.array([False, False, False, False, False, False, False, False]),
            np.array([False, False, False]),
        ]
        actual = cube.prune_indices()
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_mr_x_cat_hs_prune_indices(self):
        cube = CrunchCube(CR.MR_X_CAT_HS)
        expected = [
            np.array([False, False, False, False, False]),
            np.array([False, False, False, True, False, False, True, False]),
        ]
        actual = cube.prune_indices(transforms=[0, 1])
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_gender_x_weight_pruning(self):
        cube = CrunchCube(CR.GENDER_X_WEIGHT)
        expected = 208
        actual = cube.margin(prune=True)
        np.testing.assert_array_equal(actual, expected)

    def test_proportions_cat_x_mr_x_cat(self):
        cube = CrunchCube(CR.CAT_X_MR_X_CAT["slides"][0]["cube"])
        props = cube.proportions(axis=1, include_transforms_for_dims=[1, 2], prune=True)

        # Test first slice
        expected = np.array(
            [
                [0.3647320622384293, 0.3539601689446188],
                [0.2217369551352075, 0.21179394663351137],
                [0.2892943293978049, 0.2755799995757032],
                [0.30472468951159837, 0.31183395927247587],
                [0.1564932915102434, 0.12122561350397994],
                [0.14341569357975512, 0.16573829062583212],
                [0.3088773171830892, 0.30399159711003093],
                [0.3531835200406305, 0.4001713397700346],
                [0.2572968713520773, 0.24375187975969445],
                [0.25130939319998297, 0.1977549624477041],
                [0.274565755319545, 0.2997170882672239],
                [0.32608867343144654, 0.344478827337916],
                [0.35921238787890847, 0.3513630660099009],
                [0.25634234169007164, 0.16894820580901035],
                [0.22099299650777843, 0.22466833670878553],
                [0.2276649518247445, 0.24565762260105783],
                [0.2643381997593724, 0.1982327504994244],
                [0.41491884119922506, 0.4258666971460735],
                [0.4623019656879477, 0.520868963921971],
                [0.35277296769118416, 0.2813894529707485],
                [0.4003762375617921, 0.42681767440587187],
                [0.25925087940264385, 0.2731916153395818],
                [0.4753330440584336, 0.43648850969829933],
                [0.28148528548474344, 0.24591762645117995],
                [0.49727220036895625, 0.5108530445282087],
                [0.32590772578924143, 0.359683660082846],
                [0.3595152413326164, 0.4049397928654146],
                [0.42108775668830034, 0.3454454870295823],
                [0.4205886117048783, 0.39122538735554757],
                [0.228770284468049, 0.20059146301631123],
                [0.24603034757739972, 0.2735752782805667],
                [0.3065494749862463, 0.32314540506421324],
                [0.27194030884117687, 0.30970380251451973],
                [0.2091262494207975, 0.22920517801957993],
                [0.31769765653105564, 0.28958158962721786],
                [0.3560762345705489, 0.3102687419303191],
                [0.4804715042882989, 0.5011391743289532],
                [0.3811599892254701, 0.4226226669501276],
                [0.41029213392178276, 0.1884401015918774],
            ]
        )
        actual = compress_pruned(props[0])
        np.testing.assert_almost_equal(actual, expected)

        # Test second slice
        expected = np.array(
            [
                [0.4031214008509537, 0.4056176483118717],
                [0.24070198556407071, 0.25911741489783324],
                [0.3464300357360538, 0.33912070707907394],
                [0.3574620243397532, 0.37758989006965377],
                [0.14907456320910992, 0.15890701660953999],
                [0.20651100193920027, 0.1789634776591901],
                [0.30520247457440536, 0.3270678407142803],
                [0.3810285430516052, 0.3997309090998591],
                [0.34271319381187965, 0.17385655771473044],
                [0.3591305867483556, 0.2685531153107514],
                [0.2996020578719067, 0.29215934221779744],
                [0.3902291806673195, 0.39354067543725346],
                [0.40337866312317217, 0.4250345472210801],
                [0.32114152359818676, 0.30423847092340256],
                [0.2653311867224287, 0.27833063737964403],
                [0.313528046383254, 0.3000437600045656],
                [0.3920027268884396, 0.36933090371694116],
                [0.515781215492543, 0.4851941444303692],
                [0.5427173809468948, 0.5754703450096686],
                [0.3514753251895221, 0.3381463298582681],
                [0.40789566838261765, 0.43525540615386243],
                [0.3595684862225866, 0.38051785122829174],
                [0.556336290160639, 0.47918668411102733],
                [0.3006536550040242, 0.31770376362899333],
                [0.5132046910093269, 0.5548090283383379],
                [0.4409089206826546, 0.36616426510098665],
                [0.40390891699096854, 0.3888593109712533],
                [0.3914326716352874, 0.3346157095319079],
                [0.4423415709934932, 0.47752483308415145],
                [0.33543493750667275, 0.30087121770598385],
                [0.30409560492337334, 0.33096009035672747],
                [0.4028029720384424, 0.4497865293548307],
                [0.37991282964691514, 0.3817002730808065],
                [0.26937198075202085, 0.2530238979016483],
                [0.3367641290249356, 0.3210357156509789],
                [0.4376473666508847, 0.4044796591984694],
                [0.5986306705327854, 0.5886692367162286],
                [0.3493779725965881, 0.3533483607971598],
                [0.08962504168280223, 0.4352830423033842],
            ]
        )
        actual = compress_pruned(props[1])
        np.testing.assert_almost_equal(actual, expected)

    def test_univ_mr_with_hs_does_not_crash(self):
        """Assert that MR with H&S doesn't crash."""
        cube = CrunchCube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"])
        cube.as_array(include_transforms_for_dims=[0])
        # If it doesn't crash, the test passes, we don't actually care about
        # the result. We only care that the H&S transform doesn't crash the MR
        # variable (that it doesn't try to actually include the transform
        # in the result). H&S shouldn't be in the MR variable, but there
        # are cases when there are.
        assert True

    def test_pop_counts_ca_as_0th(self):
        cube_slice = CrunchCube(CR.CA_AS_0TH).get_slices(ca_as_0th=True)[0]
        expected = np.array(
            [54523323.46453754, 24570078.10865863, 15710358.25446403, 5072107.27712256]
        )
        actual = cube_slice.population_counts(100000000)
        np.testing.assert_almost_equal(actual, expected)

    def test_pop_counts_for_multiple_slices(self):
        cube = CrunchCube(CR.PETS_ARRAY_X_PETS)
        expected = np.array(
            [
                [
                    [24992768.29621058, 0.0, 26901938.09661558],
                    [17298235.46427536, 44258027.19120625, 21174428.69540066],
                ],
                [
                    [0.0, 19459910.91314028, 24053452.11581288],
                    [48106904.23162583, 16648106.90423161, 22216035.63474388],
                ],
                [
                    [21474773.60931439, 18272962.4838292, 0.0],
                    [25808538.16300128, 23673997.41267791, 53751617.07632601],
                ],
            ]
        )
        actual = cube.population_counts(100000000)
        np.testing.assert_almost_equal(actual, expected)
