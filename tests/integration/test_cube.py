# encoding: utf-8

"""Integration-test suite for `cr.cube.cube` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from unittest import TestCase

from cr.cube.cube import (
    Cube,
    _MeanMeasure,
    _Measures,
    _UnweightedCountMeasure,
    _WeightedCountMeasure,
)
from cr.cube.dimension import _ApparentDimensions
from cr.cube.enums import DIMENSION_TYPE as DT

from ..fixtures import CR, NA  # ---mnemonic: CR = 'cube-response'---
from ..util import load_python_expression


class DescribeIntegratedCube(object):
    """Integration-test suite for `cr.cube.cube.Cube` object."""

    def it_provides_values_for_cat_x_cat(self):
        cube = Cube(CR.CAT_X_CAT)

        assert cube.__repr__() == "Cube(name='v4', dimension_types='CAT x CAT')"
        np.testing.assert_equal(cube.counts, [[5, 2], [5, 3]])
        np.testing.assert_equal(
            cube.counts_with_missings, [[5, 3, 2, 0], [5, 2, 3, 0], [0, 0, 0, 0]]
        )
        assert cube.cube_index == 0
        assert cube.description == "Pet Owners"
        assert cube.dimension_types == (DT.CAT, DT.CAT)
        assert isinstance(cube.dimensions, _ApparentDimensions)
        assert cube.has_means is False
        assert cube.is_weighted is False
        assert cube.missing == 5
        assert cube.name == "v4"
        assert cube.ndim == 2
        assert cube.population_fraction == 1.0
        assert cube.title == "Pony Owners"
        np.testing.assert_equal(cube.unweighted_counts, [[5, 2], [5, 3]])

    @pytest.mark.parametrize(
        "cube_response, expected_dim_types",
        (
            (CR.CA_SUBVAR_HS_X_MR_X_CA_CAT, (DT.CA_SUBVAR, DT.MR, DT.CA_CAT)),
            (CR.CAT_X_LOGICAL, (DT.CAT, DT.LOGICAL)),
            (CR.LOGICAL_UNIVARIATE, (DT.LOGICAL,)),
        ),
    )
    def it_provides_access_to_its_dimensions(self, cube_response, expected_dim_types):
        cube = Cube(cube_response)

        dimension_types = tuple(d.dimension_type for d in cube.dimensions)

        assert dimension_types == expected_dim_types

    def it_provides_array_for_single_valid_cat_CAT_X_MR(self):
        """No pruning needs to happen, because pruning is based on unweighted counts:
        >>> cube.unweighted_counts
        array([[[0, 108],
                [14, 94],
                [94, 14]]])

        so even though the weighted counts are all zeroes:

        >>> cube.counts
        array([[[0, 0],
                [0, 0],
                [0, 0]]])

        we expect [[0, 0, 0]] as the result; no zero gets pruned because no unweighted
        count is zero.
        """
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_MR_SENTRY, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.counts, np.array([[0, 0, 0]]))

    def it_provides_pruned_array_for_CA_CAT_x_CA_SUBVAR(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CA_CAT_X_CA_SUBVAR, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.column_proportions,
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

    def it_provides_valid_counts_for_NUM_ARRAY_GROUPED_BY_CAT(self):
        cube = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT)

        np.testing.assert_array_equal(cube.valid_counts, [[3, 3, 1], [2, 1, 1]])

    def and_it_returns_empty_array_if_valid_counts_are_not_available(self):
        cube = Cube(CR.CAT_X_CAT)

        np.testing.assert_array_equal(cube.valid_counts, [])

    def it_provides_valid_counts_summary_for_NUM_ARRAY_GROUPED_BY_CAT(self):
        cube = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT)

        np.testing.assert_array_equal(cube.valid_counts_summary, [5, 4, 2])

    def and_it_returns_empty_array_for_summary_if_valid_counts_are_not_available(self):
        cube = Cube(CR.CAT_X_CAT)

        np.testing.assert_array_equal(cube.valid_counts_summary, [])

    def it_provides_n_responses_for_NUM_ARRAY_GROUPED_BY_CAT(self):
        cube = Cube(NA.NUM_ARR_MEANS_GROUPED_BY_CAT)

        assert cube.n_responses == 5


class DescribeIntegrated_Measures(object):
    """Integration-tests that exercise the `cr.cube.cube._Measures` object."""

    @pytest.mark.parametrize(
        "cube_response, expected_value",
        (
            # ---has {'query': {'weight': url}}---
            (CR.ADMIT_X_GENDER_WEIGHTED, True),
            # ---has {'weight_var': weight_name_str}---
            (CR.CAT_X_CAT_X_CAT_WGTD, True),
            # ---unweighted_counts == measure_count_data---
            (CR.ADMIT_X_DEPT_UNWEIGHTED, False),
            # ---has {'weight_var': weight_name_str} and no {'query':{'weight': val}}---
            (CR.CAT_X_CAT_X_CAT_WGTD_NO_QUERY_WEIGHT, True),
            # ---has {'query': {'weight_utl': url}}---
            ({"weight_url": "https://y"}, True),
        ),
    )
    def it_knows_when_its_measures_are_weighted(self, cube_response, expected_value):
        cube_dict = cube_response.get("value", cube_response)
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

    @pytest.mark.parametrize(
        "cube_dict, expected_value",
        (
            # ---filtered case---
            (CR.CAT_X_CAT_FILT, 0.254),
            # ---unfiltered case---
            (CR.CAT_X_CAT, 1.0),
            # ---complete cases---
            (CR.CAT_X_CAT_FILT_COMPLETE, 0.5760869565217391),
        ),
    )
    def it_knows_the_population_fraction(self, cube_dict, expected_value):
        measures = _Measures(cube_dict, None)

        population_fraction = measures.population_fraction

        assert population_fraction == expected_value

    def it_provides_access_to_the_unweighted_count_measure(self):
        measures = _Measures(None, None)

        unweighted_counts = measures.unweighted_counts

        assert type(unweighted_counts).__name__ == "_UnweightedCountMeasure"

    @pytest.mark.parametrize(
        "cube_dict, expected_type_name",
        (
            # ---weighted case---
            (CR.CAT_X_CAT_WGTD, "_WeightedCountMeasure"),
            # ---unweighted case---
            (CR.CAT_X_CAT, "_UnweightedCountMeasure"),
        ),
    )
    def it_provides_access_to_wgtd_count_measure(self, cube_dict, expected_type_name):
        measures = _Measures(cube_dict, None)

        weighted_counts = measures.weighted_counts

        assert type(weighted_counts).__name__ == expected_type_name

    def it_provides_access_to_the_bases_measure(self):
        slice_ = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        assert slice_.bases == (15, 15)

    def it_provides_access_to_min_base_size_mask(self):
        slice_ = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        np.testing.assert_array_equal(slice_.min_base_size_mask, [False, False])

    def it_provides_access_univariate_mr_min_base_size_mask(self):
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        np.testing.assert_array_equal(
            slice_.min_base_size_mask,
            [False, False, False, False, False, False, False, False, False],
        )

    def it_provides_access_to_its_name(self):
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        assert slice_.name == "Paid for news in the last year"

    def it_provides_access_to_its_rows_dimension_type(self):
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        assert slice_.rows_dimension_type == DT.MR

    def it_provides_access_to_its_scale_mean_when_it_is_none(self):
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        assert slice_.scale_mean is None

    def it_provides_access_to_table_name_when_it_is_ca_as_0th(self):
        slice_ = Cube(CR.CA_AS_0TH, cube_idx=0).partitions[0]
        assert slice_.table_name == "Level of interest: ATP Men's Tennis"

    def it_knows_unweighted_bases(self):
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        assert slice_.unweighted_bases == (33358,) * 9


class DescribeIntegrated_MeanMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT_MEAN_WGTD
        cube = Cube(cube_dict)
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

    def it_handles_cat_x_mr_with_means(self):
        slice_ = Cube(CR.MEANS_CAT_X_MR).partitions[0]
        assert slice_.column_labels.tolist() == [
            "Denmark",
            "Finland",
            "Iceland",
            "Norway",
            "Sweden",
        ]

    def it_handles_means_cat_hs_x_cat_hs(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]

        means = slice_.means

        np.testing.assert_array_almost_equal(
            means,
            [
                [41.96875, 30.875, 25.66666667, np.nan, 42.0],
                [51.51515152, 47.95555556, 45.44444444, np.nan, 45.0952381],
                [46.17088608, 44.55504587, 48.09090909, np.nan, 50.8],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [44.03030303, 45.21568627, 54.53333333, np.nan, 56.19512195],
                [45.64516129, 47.41428571, 46.89361702, np.nan, 55.27894737],
                [34.20408163, 43.2745098, 41.2, np.nan, 35.26086957],
            ],
        )

    def it_knows_if_it_has_means(self):
        slice_ = Cube(CR.MEANS_CAT_HS_X_CAT_HS).partitions[0]
        assert slice_.has_means


class DescribeIntegrated_UnweightedCountMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT
        cube = Cube(cube_dict)
        measure = _UnweightedCountMeasure(cube_dict, cube._all_dimensions)

        raw_cube_array = measure.raw_cube_array

        np.testing.assert_array_almost_equal(
            raw_cube_array, [[5, 3, 2, 0], [5, 2, 3, 0], [0, 0, 0, 0]]
        )


class DescribeIntegrated_WeightedCountMeasure(object):
    def it_provides_access_to_its_raw_cube_array(self):
        cube_dict = CR.CAT_X_CAT_WGTD
        cube = Cube(cube_dict)
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


class TestCrunchCubeAsNub(TestCase):
    """Legacy unit-tests for 0D cube."""

    def test_mean_no_dims(self):
        cube = Cube(CR.ECON_MEAN_NO_DIMS)
        assert cube.description is None
        assert cube.name is None
        assert cube.missing == 0

        nub = cube.partitions[0]

        np.testing.assert_almost_equal(nub.means, np.array([49.095]))
        assert nub.ndim == 0
        np.testing.assert_almost_equal(nub.table_base, np.array([49.095]))
        np.testing.assert_array_equal(nub.unweighted_count, 1000)


class TestCrunchCubeAs_Slice(object):
    """ ... """

    def test_crunch_cube_loads_data(self):
        cube = Cube(CR.CAT_X_CAT)
        cube_dict = cube._cube_dict
        assert cube_dict == CR.CAT_X_CAT

    def test_as_array_univariate_cat_exclude_missing(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        assert strand.counts == (10, 5)
        assert strand.unweighted_counts == (10, 5)

    def test_as_array_numeric(self):
        strand = Cube(CR.VOTER_REGISTRATION).partitions[0]
        assert strand.counts == (885, 105, 10)

    def test_as_array_datetime(self):
        strand = Cube(CR.SIMPLE_DATETIME).partitions[0]
        assert strand.counts == (1, 1, 1, 1)

    def test_as_array_text(self):
        strand = Cube(CR.SIMPLE_TEXT).partitions[0]
        assert strand.counts == (1, 1, 1, 1, 1, 1)

    def test_as_array_cat_x_cat_exclude_missing(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[5, 2], [5, 3]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_as_array_cat_x_cat_unweighted(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[5, 2], [5, 3]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_as_array_cat_x_datetime_exclude_missing(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_margin_univariate_cat_axis_none(self):
        slice_ = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        expected = np.array([15])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_margin_numeric(self):
        slice_ = Cube(CR.VOTER_REGISTRATION).partitions[0]
        expected = np.array([1000])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_margin_datetime(self):
        slice_ = Cube(CR.SIMPLE_DATETIME).partitions[0]
        expected = np.array([4])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_margin_text(self):
        slice_ = Cube(CR.SIMPLE_TEXT).partitions[0]
        expected = np.array([6])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_cat_x_cat_table_margin(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([15])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_mr_x_mr_table_base_and_margin(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.MR_X_MR_WITH_PRUNING, transforms=transforms).partitions[0]

        # Assert table base
        expected = np.full((10, 10), 6490)
        np.testing.assert_array_equal(slice_.table_base, expected)

        # Assert table margin
        expected = np.full((10, 10), 6456.761929)
        np.testing.assert_almost_equal(slice_.table_margin, expected)

    def test_mr_x_mr_table_base_and_margin_unpruned(self):
        slice_ = Cube(CR.MR_X_MR_WITH_PRUNING).partitions[0]

        # Assert table base
        expected = np.full((12, 12), 6490)
        np.testing.assert_array_equal(slice_.table_base_unpruned, expected)

        # Assert table margin
        expected = np.full((12, 12), 6456.761929)
        np.testing.assert_almost_equal(slice_.table_margin_unpruned, expected)

    def test_margin_cat_x_datetime_axis_none(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array([4])
        np.testing.assert_array_equal(slice_.table_margin, expected)

    def test_margin_cat_x_cat_axis_0(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([10, 5])
        np.testing.assert_array_equal(slice_.columns_margin, expected)

    def test_margin_cat_x_datetime_axis_0(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array([1, 1, 1, 1])
        np.testing.assert_array_equal(slice_.columns_margin, expected)

    def test_margin_cat_x_cat_axis_1(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([7, 8])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_margin_cat_x_datetime_axis_1(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array([1, 1, 1, 1, 0])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_proportions_univariate_cat_axis_none(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        np.testing.assert_almost_equal(strand.table_proportions, (0.6666667, 0.3333333))

    def test_proportions_numeric(self):
        strand = Cube(CR.VOTER_REGISTRATION).partitions[0]
        table_proportions = strand.table_proportions
        np.testing.assert_almost_equal(table_proportions, (0.885, 0.105, 0.010))

    def test_proportions_datetime(self):
        strand = Cube(CR.SIMPLE_DATETIME).partitions[0]
        table_proportions = strand.table_proportions
        np.testing.assert_almost_equal(table_proportions, (0.25, 0.25, 0.25, 0.25))

    def test_proportions_text(self):
        strand = Cube(CR.SIMPLE_TEXT).partitions[0]
        table_proportions = strand.table_proportions
        np.testing.assert_almost_equal(
            table_proportions,
            (0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667, 0.1666667),
        )

    def test_std_dev_err_moe_univariate_cat_axis_none(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL, population=1000).partitions[0]
        np.testing.assert_almost_equal(
            strand.standard_deviation, [0.47140452, 0.47140452]
        )
        np.testing.assert_almost_equal(strand.standard_error, [0.1217161, 0.1217161])
        np.testing.assert_almost_equal(
            strand.table_proportions_moe, [0.238559221, 0.238559221]
        )
        np.testing.assert_almost_equal(
            strand.population_moe, [238.55922104, 238.55922104]
        )

    def test_std_dev_err_numeric(self):
        strand = Cube(CR.VOTER_REGISTRATION, population=1000).partitions[0]
        np.testing.assert_almost_equal(
            strand.standard_deviation, [0.31902194, 0.30655342, 0.09949874]
        )
        np.testing.assert_almost_equal(
            strand.standard_error, [0.0100884, 0.0096941, 0.0031464]
        )
        np.testing.assert_almost_equal(
            strand.table_proportions_moe, [0.019772822, 0.019000029, 0.006166883]
        )
        np.testing.assert_almost_equal(
            strand.population_moe, [19.77282169, 19.0000289, 6.16688276]
        )

    def test_std_dev_err_datetime(self):
        strand = Cube(CR.SIMPLE_DATETIME).partitions[0]
        np.testing.assert_almost_equal(
            strand.standard_deviation, [0.4330127, 0.4330127, 0.4330127, 0.4330127]
        )
        np.testing.assert_almost_equal(
            strand.standard_error, [0.2165064, 0.2165064, 0.2165064, 0.2165064]
        )

    def test_std_dev_err_text(self):
        strand = Cube(CR.SIMPLE_TEXT).partitions[0]
        np.testing.assert_almost_equal(
            strand.standard_deviation,
            [0.372678, 0.372678, 0.372678, 0.372678, 0.372678, 0.372678],
        )
        np.testing.assert_almost_equal(
            strand.standard_error,
            [0.1521452, 0.1521452, 0.1521452, 0.1521452, 0.1521452, 0.1521452],
        )

    def test_proportions_cat_x_cat_axis_none(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[0.3333333, 0.1333333], [0.3333333, 0.2000000]])
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_proportions_cat_x_datetime_axis_none(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array(
            [
                [0.0, 0.0, 0.25, 0.0],
                [0.0, 0.0, 0.0, 0.25],
                [0.0, 0.25, 0.0, 0.0],
                [0.25, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_proportions_cat_x_cat_axis_0(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[0.5, 0.4], [0.5, 0.6]])
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_proportions_cat_x_datetime_axis_0(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array(
            [[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_proportions_cat_x_cat_axis_1(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[0.7142857, 0.2857143], [0.6250000, 0.3750000]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_proportions_cat_x_datetime_axis_1(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        expected = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [np.nan, np.nan, np.nan, np.nan],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_percentages_univariate_cat_axis_none(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL).partitions[0]
        table_percentages = strand.table_percentages
        np.testing.assert_almost_equal(table_percentages, (66.6666667, 33.3333333))

    def test_percentages_numeric(self):
        strand = Cube(CR.VOTER_REGISTRATION).partitions[0]
        table_percentages = strand.table_percentages
        np.testing.assert_almost_equal(table_percentages, (88.5, 10.5, 1.0))

    def test_percentages_datetime(self):
        strand = Cube(CR.SIMPLE_DATETIME).partitions[0]
        table_percentages = strand.table_percentages
        np.testing.assert_almost_equal(table_percentages, (25.0, 25.0, 25.0, 25.0))

    def test_percentages_text(self):
        strand = Cube(CR.SIMPLE_TEXT).partitions[0]
        table_percentages = strand.table_percentages
        np.testing.assert_almost_equal(
            table_percentages,
            (16.6666667, 16.6666667, 16.6666667, 16.6666667, 16.6666667, 16.6666667),
        )

    def test_percentages_cat_x_cat_axis_none(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[33.3333333, 13.3333333], [33.3333333, 20.0]])
        np.testing.assert_almost_equal(slice_.table_percentages, expected)

    def test_percentages_cat_x_cat_axis_0(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[50, 40], [50, 60]])
        np.testing.assert_almost_equal(slice_.column_percentages, expected)

    def test_percentages_cat_x_cat_axis_1(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        expected = np.array([[71.4285714, 28.5714286], [62.50000, 37.50000]])
        np.testing.assert_almost_equal(slice_.row_percentages, expected)

    def test_population_counts_univariate_cat(self):
        strand = Cube(CR.UNIVARIATE_CATEGORICAL, population=9001).partitions[0]
        population_counts = strand.population_counts
        np.testing.assert_almost_equal(population_counts, (6000.6666667, 3000.3333333))

    def test_population_counts_numeric(self):
        strand = Cube(CR.VOTER_REGISTRATION, population=9001).partitions[0]
        population_counts = strand.population_counts
        np.testing.assert_almost_equal(population_counts, (7965.885, 945.105, 90.01))

    def test_population_counts_datetime(self):
        strand = Cube(CR.SIMPLE_DATETIME, population=9001).partitions[0]
        population_counts = strand.population_counts
        np.testing.assert_almost_equal(
            population_counts, (2250.25, 2250.25, 2250.25, 2250.25)
        )

    def test_population_counts_text(self):
        strand = Cube(CR.SIMPLE_TEXT, population=9001).partitions[0]
        population_counts = strand.population_counts
        np.testing.assert_almost_equal(
            population_counts,
            (
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
                1500.1666667,
            ),
        )

    def test_population_counts_cat_x_cat(self):
        slice_ = Cube(CR.CAT_X_CAT, population=9001).partitions[0]
        expected = np.array([[3000.3333333, 1200.1333333], [3000.3333333, 1800.2]])
        np.testing.assert_almost_equal(slice_.population_counts, expected)

    def test_filtered_population_counts(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.CAT_X_CAT_FILT, transforms=transforms, population=100000000
        ).partitions[0]
        expected = np.array(
            [
                [300000.0, 1400000.0, 0.0, 0.0, 0.0, 0.0],
                [5900000.0, 13200000.0, 0.0, 0.0, 0.0, 0.0],
                [600000.0, 2900000.0, 0.0, 0.0, 0.0, 0.0],
                [100000.0, 100000.0, 0.0, 0.0, 0.0, 0.0],
                [300000.0, 600000.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.population_counts, expected)

    def test_labels_cat_x_cat_exclude_missing(self):
        slice_ = Cube(CR.CAT_X_CAT).partitions[0]
        assert slice_.row_labels.tolist() == ["B", "C"]
        assert slice_.column_labels.tolist() == ["C", "E"]

    def test_labels_cat_x_datetime_exclude_missing(self):
        slice_ = Cube(CR.CAT_X_DATETIME).partitions[0]
        assert slice_.row_labels.tolist() == ["red", "green", "blue", "4", "9"]
        assert slice_.column_labels.tolist() == [
            "1776-07-04T00:00:00",
            "1950-12-24T00:00:00",
            "2000-01-01T00:00:00",
            "2000-01-02T00:00:00",
        ]

    def test_labels_simple_cat_array_exclude_missing(self):
        slice_ = Cube(CR.SIMPLE_CAT_ARRAY).partitions[0]
        assert slice_.row_labels.tolist() == [
            "ca_subvar_1",
            "ca_subvar_2",
            "ca_subvar_3",
        ]
        assert slice_.column_labels.tolist() == ["a", "b", "c", "d"]

    def test_as_array_simple_cat_array_exclude_missing(self):
        slice_ = Cube(CR.SIMPLE_CAT_ARRAY).partitions[0]
        expected = np.array([[3, 3, 0, 0], [1, 3, 2, 0], [0, 2, 1, 3]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_as_array_cat_x_num_x_datetime(self):
        """Test 3D cube, slicing accross first (numerical) variable."""
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[1, 1], [0, 0], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.counts, [[2, 1], [1, 1], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [2, 3], [0, 0], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [0, 0], [3, 2], [0, 0]])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.counts, [[0, 0], [0, 0], [1, 1], [0, 1]])

    def test_proportions_cat_x_num_datetime(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.4, 0.2], [0.2, 0.2], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.0, 0.0], [0.4, 0.6], [0.0, 0.0], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_almost_equal(
            slice_.table_proportions, [[0.0, 0.0], [0.0, 0.0], [0.6, 0.4], [0.0, 0.0]]
        )
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_almost_equal(
            slice_.table_proportions,
            [[0.0, 0.0], [0.0, 0.0], [0.33333333, 0.33333333], [0.0, 0.33333333]],
        )

    def test_cat_x_num_x_datetime_margin_by_table(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, 2)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.table_margin, 5)
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.table_margin, 3)

    def test_margin_cat_x_num_x_datetime_axis_0(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.rows_margin, [2, 0, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.rows_margin, [3, 2, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 5, 0, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 0, 5, 0])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.rows_margin, [0, 0, 2, 1])

    def test_margin_cat_x_num_x_datetime_axis_1(self):
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 1])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[1]
        np.testing.assert_array_equal(slice_.columns_margin, [3, 2])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[2]
        np.testing.assert_array_equal(slice_.columns_margin, [2, 3])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[3]
        np.testing.assert_array_equal(slice_.columns_margin, [3, 2])
        slice_ = Cube(CR.CAT_X_NUM_X_DATETIME).partitions[4]
        np.testing.assert_array_equal(slice_.columns_margin, [1, 2])

    def test_as_array_unweighted_gender_x_ideology(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([[32, 85, 171, 114, 70, 13], [40, 97, 205, 106, 40, 27]])
        np.testing.assert_array_equal(slice_.unweighted_counts, expected)

    def test_as_array_weighted_gender_x_ideology(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
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
        np.testing.assert_almost_equal(slice_.counts, expected)

    def test_margin_weighted_gender_x_ideology_axis_0(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
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
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_margin_unweighted_gender_x_ideology_axis_0(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([72, 182, 376, 220, 110, 40])
        np.testing.assert_array_equal(slice_.columns_base, expected)

    def test_margin_unweighted_gender_x_ideology_axis_1(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([485, 515])
        np.testing.assert_array_equal(slice_.rows_base, expected)

    def test_margin_weighted_gender_x_ideology_axis_1(self):
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected = np.array([500, 500])
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_calculate_various_measures_axis_0(self):
        """Calculate standard error across columns."""
        slice_ = Cube(CR.ECON_GENDER_X_IDEOLOGY_WEIGHTED).partitions[0]
        expected_zscore = np.array(
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
        expected_table_std_dev = [
            [0.17860955, 0.28275439, 0.38106557, 0.32204575, 0.25876083, 0.1149889],
            [0.19320144, 0.29207169, 0.39927, 0.30384472, 0.19320144, 0.15976996],
        ]
        expected_table_std_err = [
            [0.00564813, 0.00894148, 0.01205035, 0.01018398, 0.00818274, 0.00363627],
            [0.00610957, 0.00923612, 0.01262603, 0.00960841, 0.00610957, 0.00505237],
        ]
        expected_col_std_dev = [
            [0.49834148, 0.4996758, 0.49908137, 0.49890016, 0.47692704, 0.47313192],
            [0.49834148, 0.4996758, 0.49908137, 0.49890016, 0.47692704, 0.47313192],
        ]
        expected_col_std_err = [
            [0.05880176, 0.03705843, 0.02576154, 0.03360238, 0.04526793, 0.07517074],
            [0.05880176, 0.03705843, 0.02576154, 0.03360238, 0.04526793, 0.07517074],
        ]

        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-col-prop-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.row_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-row-prop-moe"),
        )
        np.testing.assert_almost_equal(
            slice_.row_std_dev,
            load_python_expression("econ-gender-x-ideology-weighted-row-std-dev"),
        )
        np.testing.assert_almost_equal(
            slice_.table_proportions_moe,
            load_python_expression("econ-gender-x-ideology-weighted-table-prop-moe"),
        )
        np.testing.assert_almost_equal(slice_.zscores, expected_zscore)

    def test_pvals(self):
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
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning and H&S
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_X_CAT_GERMAN_WEIGHTED, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_pvals_stats(self):
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
        slice_ = Cube(CR.STATS_TEST).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.STATS_TEST, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

        # Test with pruning and H&S
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.STATS_TEST, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(slice_.pvals, expected)

    def test_mean_age_for_blame_x_gender(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(
            CR.ECON_MEAN_AGE_BLAME_X_GENDER, transforms=transforms
        ).partitions[0]
        expected = np.array(
            [
                [52.78205128205122, 49.9069767441861],
                [50.43654822335009, 48.20100502512572],
                [51.5643564356436, 47.602836879432715],
                [58, 29],
                [37.53846153846155, 39.45238095238095],
            ]
        )
        np.testing.assert_almost_equal(slice_.means, expected)
        assert slice_.ndim == 2

    def test_various_measures_admit_by_dept_unweighted_rows(self):
        """see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/
        zvalues-spec.js#L42
        """
        slice_ = Cube(CR.ADMIT_X_DEPT_UNWEIGHTED).partitions[0]
        expected_zscores = np.array(
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
        expected_table_std_dev = [
            [0.33934583, 0.27398329, 0.25706606, 0.2364359, 0.17726851, 0.10030056],
            [0.26071661, 0.21271283, 0.33814647, 0.31969003, 0.29534847, 0.35469478],
        ]
        expected_table_std_err = [
            [0.00504412, 0.00407255, 0.00382109, 0.00351444, 0.00263496, 0.00149089],
            [0.00387535, 0.00316181, 0.00502629, 0.00475195, 0.00439013, 0.00527227],
        ]
        expected_col_std_dev = [
            [0.47876747, 0.48213008, 0.47720873, 0.47358921, 0.43399681, 0.24550986],
            [0.47876747, 0.48213008, 0.47720873, 0.47358921, 0.43399681, 0.24550986],
        ]
        expected_col_std_err = [
            [0.01567414, 0.01993363, 0.01575024, 0.01682826, 0.01795892, 0.00918798],
            [0.01567414, 0.01993363, 0.01575024, 0.01682826, 0.01795892, 0.00918798],
        ]

        np.testing.assert_almost_equal(slice_.zscores, expected_zscores)
        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe,
            load_python_expression("admit-x-dept-unweighted-col-prop-moe"),
        )

    def test_various_measures_admit_by_gender_weighted_rows(self):
        """see
        https://github.com/Crunch-io/whaam/blob/master/base/stats/tests/
        zvalues-spec.js#L67
        """
        slice_ = Cube(CR.ADMIT_X_GENDER_WEIGHTED).partitions[0]
        expected_zscores = np.array(
            [
                [9.42561984520692, -9.425619845206922],
                [-9.425619845206922, 9.42561984520692],
            ]
        )
        expected_table_std_dev = [[0.44013199, 0.32828883], [0.47059018, 0.45061221]]
        expected_table_std_err = [[0.00659641, 0.00492018], [0.0070529, 0.00675348]]
        expected_col_std_dev = [[0.49668253, 0.45933735], [0.49668253, 0.45933735]]
        expected_col_std_err = [[0.00966009, 0.01080163], [0.00966009, 0.01080163]]
        expected_col_proportions_moe = [
            [0.0189334366, 0.0211708092],
            [0.0189334366, 0.0211708092],
        ]

        np.testing.assert_almost_equal(slice_.zscores, expected_zscores)
        np.testing.assert_almost_equal(slice_.table_std_dev, expected_table_std_dev)
        np.testing.assert_almost_equal(slice_.table_std_err, expected_table_std_err)
        np.testing.assert_almost_equal(slice_.column_std_dev, expected_col_std_dev)
        np.testing.assert_almost_equal(slice_.column_std_err, expected_col_std_err)
        np.testing.assert_almost_equal(
            slice_.column_proportions_moe, expected_col_proportions_moe
        )

    def test_selected_crosstab_as_array(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.counts, expected)

    def test_selected_crosstab_margin_by_rows(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.rows_margin, expected)

    def test_selected_crosstab_margin_by_cols(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.columns_margin, expected)

    def test_selected_crosstab_margin_total(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.table_margin, expected)

    def test_selected_crosstab_proportions_by_rows(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_selected_crosstab_proportions_by_cols(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_selected_crosstab_proportions_by_cell(self):
        slice_ = Cube(CR.MR_X_CAT_2).partitions[0]
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
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_as_array(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array([[40, 14, 18], [14, 34, 16], [18, 16, 38]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_x_pets_proportions_by_cell(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [0.5, 0.2, 0.2571429],
                [0.2, 0.4303797, 0.2285714],
                [0.2571429, 0.2285714, 0.5428571],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_proportions_by_col(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [1.0, 0.4827586, 0.4736842],
                [0.4117647, 1.0, 0.4210526],
                [0.5294118, 0.5517241, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_pets_proportions_by_row(self):
        slice_ = Cube(CR.PETS_X_PETS).partitions[0]
        expected = np.array(
            [
                [1.0, 0.4117647, 0.5294118],
                [0.4827586, 1.0, 0.5517241],
                [0.4736842, 0.4210526, 1.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_x_fruit_as_array(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array([[12, 28], [12, 22], [12, 26]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_x_fruit_margin_row(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array([40, 34, 38])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_pets_array_as_array(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array([[45, 34], [40, 40], [32, 38]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_pets_array_proportions(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array(
            [[0.5696203, 0.4303797], [0.5000000, 0.500000], [0.4571429, 0.5428571]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_array_margin_by_row(self):
        slice_ = Cube(CR.PETS_ARRAY).partitions[0]
        expected = np.array([79, 80, 70])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_fruit_x_pets_proportions_by_cell(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array(
            [[0.15, 0.15189873, 0.17142857], [0.35, 0.27848101, 0.37142857]]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_fruit_x_pets_proportions_by_row(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array(
            [[0.4285714, 0.48, 0.5217391], [0.5384615, 0.4074074, 0.5531915]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_fruit_x_pets_proportions_by_col(self):
        slice_ = Cube(CR.FRUIT_X_PETS).partitions[0]
        expected = np.array([[0.3, 0.3529412, 0.3157895], [0.7, 0.6470588, 0.6842105]])
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_fruit_proportions_by_cell(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.15, 0.35], [0.15189873, 0.27848101], [0.17142857, 0.37142857]]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_fruit_proportions_by_col(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.4285714, 0.5384615], [0.48, 0.4074074], [0.5217391, 0.5531915]]
        )
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_x_fruit_proportions_by_row(self):
        slice_ = Cube(CR.PETS_X_FRUIT).partitions[0]
        expected = np.array(
            [[0.3, 0.7], [0.3529412, 0.6470588], [0.3157895, 0.6842105]]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_cat_x_cat_array_proportions_by_row(self):
        """Get the proportions for each slice of the 3D cube."""
        slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[0]
        expected = [[0.52, 0.48], [0.57142857, 0.42857143], [0.47826087, 0.52173913]]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.FRUIT_X_PETS_ARRAY).partitions[1]
        expected = [
            [0.59259259, 0.40740741],
            [0.46153846, 0.53846154],
            [0.44680851, 0.55319149],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_identity_x_period_axis_out_of_bounds(self):
        slice_ = Cube(CR.NUM_X_NUM_EMPTY).partitions[0]
        expected = np.array([94, 0, 248, 210, 102, 0, 0, 0, 286, 60])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_ca_with_single_cat(self):
        slice_ = Cube(CR.CA_SINGLE_CAT).partitions[0]
        expected = np.array([79, 80, 70, 0])
        np.testing.assert_almost_equal(slice_.rows_base, expected)

    def test_pets_array_x_pets_by_col(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = [0.59097127, 0.0, 0.55956679], [0.40902873, 1.0, 0.44043321]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_pets_array_x_pets_row(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = [0.44836533, 0.0, 0.48261546], [0.39084967, 1.0, 0.47843137]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_pets_array_x_pets_cell(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS).partitions[0]
        expected = (
            [0.24992768, 0.00000000, 0.26901938],
            [0.17298235, 0.44258027, 0.21174429],
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_pets_x_pets_array_percentages(self):
        slice_ = Cube(CR.PETS_X_PETS_ARRAY).partitions[0]
        expected = [
            [0.58823529, 0.41176471],
            [0.00000000, 1.00000000],
            [0.47058824, 0.52941176],
        ]
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_profiles_percentages_add_up_to_100(self):
        slice_ = Cube(CR.PROFILES_PERCENTS).partitions[0]
        props = slice_.row_percentages
        actual_sum = np.sum(props, axis=1)
        expected_sum = np.ones(props.shape[0]) * 100
        np.testing.assert_almost_equal(actual_sum, expected_sum)

    def test_cat_x_cat_as_array_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
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
        np.testing.assert_array_equal(slice_.counts, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array([[2, 2, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 1, 0]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_cat_x_cat_props_by_col_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
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
        np.testing.assert_array_equal(slice_.column_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [1.0, 0.25, 0.25],
                [0.0, 0.125, 0.5],
                [0.0, 0.25, 0.0],
                [0.0, 0.25, 0.25],
                [0.0, 0.125, 0.0],
            ]
        )
        np.testing.assert_array_equal(slice_.column_proportions, expected)

    def test_cat_x_cat_props_by_row_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
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
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [0.4, 0.4, 0.2],
                [0.0, 0.33333333, 0.66666667],
                [0.0, 1.0, 0.0],
                [0.0, 0.66666667, 0.33333333],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_cat_x_cat_props_by_cell_prune_cols(self):
        # No pruning
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS).partitions[0]
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
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

        # With pruning
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(CR.CAT_X_CAT_WITH_EMPTY_COLS, transforms=transforms).partitions[0]
        expected = np.array(
            [
                [0.14285714, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.14285714],
                [0.0, 0.14285714, 0.0],
                [0.0, 0.14285714, 0.07142857],
                [0.0, 0.07142857, 0.0],
            ]
        )
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_prune_univariate_cat(self):
        transforms = {"rows_dimension": {"prune": True}}
        strand = Cube(CR.BINNED, transforms=transforms).partitions[0]

        counts = strand.counts

        np.testing.assert_almost_equal(
            counts, (118504.40402204, 155261.2723631, 182923.95470245)
        )

    def test_single_col_margin_not_iterable(self):
        slice_ = Cube(CR.SINGLE_COL_MARGIN_NOT_ITERABLE).partitions[0]
        assert slice_.columns_margin == 1634

    def test_3d_percentages_by_col(self):
        # ---CAT x CAT x CAT---
        slice_ = Cube(CR.GENDER_PARTY_RACE).partitions[0]
        expected = [
            [0.17647059, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.17647059, 0.05882353, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.23529412, 0.0, 0.0, 0.0, 0.0, 0.05882353, 0.0, 0.0],
            [0.11764706, 0.05882353, 0.0, 0.05882353, 0.0, 0.05882353, 0.0, 0.0],
        ]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

        expected = [
            [0.04761905, 0.0, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0],
            [0.14285714, 0.04761905, 0.0952381, 0.04761905, 0.0, 0.04761905, 0.0, 0.0],
            [0.23809524, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.19047619, 0.0, 0.04761905, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        slice_ = Cube(CR.GENDER_PARTY_RACE).partitions[1]
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_total_unweighted_margin_when_has_means(self):
        """Tests that total margin is Unweighted N, when cube has means."""
        strand = Cube(CR.CAT_MEAN_WGTD).partitions[0]
        assert len(strand.means) == 6367
        assert strand.table_margin == 17615
        assert strand.ndim == 1
        assert strand.shape == (6367,)

    def test_1D_means_pruned(self):
        """Tests that total margin is Unweighted N, when cube has means."""
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CAT_MEAN_WGTD, transforms=transforms).partitions[0]
        np.testing.assert_almost_equal(
            slice_.means,
            [
                74.50346622,
                83.41506223,
                83.82949734,
                80.44269781,
                79.38796217,
                68.23023886,
                77.02288763,
                78.86549638,
                85.31353177,
                83.61415346,
                83.0,
                65.89693441,
                61.7454857,
                83.28116575,
                82.88105494,
                84.12529155,
                71.36926833,
                77.92065684,
                77.29306375,
                79.73236992,
                84.49053524,
                67.86766862,
                72.94684356,
                82.56469913,
                71.05086738,
                83.85586938,
                81.74092388,
                57.6486009,
                82.98626745,
                78.11554074,
                76.16142377,
                76.43885371,
                71.04143565,
                82.79687964,
                54.20533614,
            ],
        )
        assert slice_.table_base == 17615

    def test_row_unweighted_margin_when_has_means(self):
        # TODO: Fix after base is implemented for means partitions
        """Tests that total margin is Unweighted N, when cube has means."""
        transforms = {"rows_dimension": {"prune": True}}
        strand = Cube(CR.CAT_MEAN_WGTD, transforms=transforms).partitions[0]
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
        np.testing.assert_array_equal(strand.rows_base, expected)
        # --- not testing cube.prune_indices() because the margin has 6367 cells ---

    def test_ca_with_single_cat_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CA_SINGLE_CAT, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.unweighted_counts, [[79], [80], [70]])

    def test_ca_x_single_cat_counts(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        expected = [[13], [12]]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        expected = [[16], [12]]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        expected = [[11], [12]]
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_ca_x_single_cat_props_by_col(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        expected = [[0.52], [0.48]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        expected = [[0.57142857], [0.42857143]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        expected = [[0.47826087], [0.52173913]]
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_ca_x_single_cat_props_by_row(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        expected = np.array([[1.0], [1.0]])
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_ca_x_single_cat_props_by_cell(self):
        expected = [
            [[0.52], [0.48]],
            [[0.57142857], [0.42857143]],
            [[0.47826087], [0.52173913]],
        ]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_almost_equal(slice_.table_proportions, expected[2])

    def test_ca_x_single_cat_col_margins(self):
        expected = [25, 28, 23]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.columns_margin, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.columns_margin, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.columns_margin, expected[2])

    def test_ca_x_single_cat_rows_margins(self):
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.rows_margin, [13, 12])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.rows_margin, [16, 12])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.rows_margin, [11, 12])

    def test_ca_x_single_cat_cell_margins(self):
        expected = [25, 28, 23]
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, expected[0])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[1]
        np.testing.assert_array_equal(slice_.table_margin, expected[1])
        slice_ = Cube(CR.CA_X_SINGLE_CAT).partitions[2]
        np.testing.assert_array_equal(slice_.table_margin, expected[2])

    def test_ca_subvar_x_cat_hs_counts_prune(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.CA_SUBVAR_HS_X_CAT_HS, transforms=transforms).partitions[0]
        expected = np.array([[3, 3, 0, 0, 6], [1, 3, 2, 0, 4], [0, 2, 1, 3, 2]])
        np.testing.assert_array_equal(slice_.counts, expected)

    def test_values_services(self):
        slice_ = Cube(CR.MR_X_CA_CAT_X_CA_SUBVAR).partitions[0]
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
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_mr_props_with_hs_by_cell(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
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
        np.testing.assert_almost_equal(slice_.table_proportions, expected)

    def test_mr_props_with_hs_by_row(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
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
        np.testing.assert_almost_equal(slice_.row_proportions, expected)

    def test_mr_props_with_hs_by_col(self):
        slice_ = Cube(CR.LETTERS_X_PETS_HS).partitions[0]
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
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_3d_pruning_indices(self):
        """Test pruning indices for a simple XYZ cube."""
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

        # Both rows and columns get pruned
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[1]])

        # Just columns get pruned
        transforms = {"columns_dimension": {"insertions": {}, "prune": True}}
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts, [[1], [0], [0], [1], [1], [0], [0], [1]]
        )

        # Just rows get pruned
        transforms = {"rows_dimension": {"insertions": {}, "prune": True}}
        slice_ = Cube(CR.XYZ_SIMPLE_ALLTYPES, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.counts, [[0, 1, 0]])

    def test_mr_x_ca_rows_margin(self):
        slice_ = Cube(CR.MR_X_CA_HS).partitions[0]
        expected = np.array([3, 3, 3])
        np.testing.assert_array_equal(slice_.rows_margin, expected)
        slice_ = Cube(CR.MR_X_CA_HS).partitions[1]
        expected = np.array([4, 4, 4])
        np.testing.assert_array_equal(slice_.rows_margin, expected)
        slice_ = Cube(CR.MR_X_CA_HS).partitions[2]
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(slice_.rows_margin, expected)

    def test_ca_x_mr_margin(self):
        slice_ = Cube(CR.CA_X_MR_WEIGHTED_HS).partitions[0]
        expected = np.array([504, 215, 224, 76, 8, 439])
        np.testing.assert_array_equal(slice_.columns_base, expected)

    def test_ca_x_mr_margin_prune(self):
        # ---CA x MR---
        transforms = {
            "rows_dimension": {"insertions": {}, "prune": True},
            "columns_dimension": {"insertions": {}, "prune": True},
        }
        slice_ = Cube(CR.CA_X_MR_WEIGHTED_HS, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(
            slice_.columns_base, np.array([504, 215, 224, 76, 8, 439])
        )
        assert slice_.table_name == "q1. Aftensmad: K\xf8d (svin/lam/okse)"

    def test_mr_x_cat_x_mr_pruning(self):
        # No pruning
        slice_ = Cube(CR.MR_X_CAT_X_MR_PRUNE).partitions[0]
        np.testing.assert_array_equal(
            slice_.counts,
            [
                [9, 7, 5, 0],
                [0, 5, 2, 0],
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )

        # With pruning
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.MR_X_CAT_X_MR_PRUNE, transforms=transforms).partitions[0]
        # Last column is not pruned, because the not-selected base counts
        # (for that column) are not all zeros.
        np.testing.assert_array_equal(
            slice_.counts,
            [[9, 7, 5, 0], [0, 5, 2, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]],
        )

    def test_gender_x_weight_pruning(self):
        transforms = {"rows_dimension": {"prune": True}}
        slice_ = Cube(CR.GENDER_X_WEIGHT, transforms=transforms).partitions[0]
        np.testing.assert_array_equal(slice_.table_margin, 208)

    def test_proportions_cat_x_mr_x_cat(self):
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(
            CR.CAT_X_MR_X_CAT["slides"][0]["cube"], transforms=transforms
        ).partitions[0]

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
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

        # Test second slice
        transforms = {
            "rows_dimension": {"prune": True},
            "columns_dimension": {"prune": True},
        }
        slice_ = Cube(
            CR.CAT_X_MR_X_CAT["slides"][0]["cube"], transforms=transforms
        ).partitions[1]
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
        np.testing.assert_almost_equal(slice_.column_proportions, expected)

    def test_univ_mr_with_hs_does_not_crash(self):
        """Assert that MR with H&S doesn't crash."""
        slice_ = Cube(CR.UNIV_MR_WITH_HS["slides"][0]["cube"]).partitions[0]
        slice_.counts
        # If it doesn't crash, the test passes, we don't actually care about
        # the result. We only care that the H&S transform doesn't crash the MR
        # variable (that it doesn't try to actually include the transform
        # in the result). H&S shouldn't be in the MR variable, but there
        # are cases when there are.
        assert True

    def test_pop_counts_ca_as_0th(self):
        transforms = {
            "columns_dimension": {"insertions": {}},
            "rows_dimension": {"insertions": {}},
        }
        slice_ = Cube(
            CR.CA_AS_0TH, cube_idx=0, transforms=transforms, population=100000000
        ).partitions[0]

        population_counts = slice_.population_counts

        np.testing.assert_almost_equal(
            population_counts,
            (54523323.46453754, 24570078.10865863, 15710358.25446403, 5072107.27712256),
        )

    def test_partitions(self):
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[0]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [24992768.29621058, 0.0, 26901938.09661558],
                [17298235.46427536, 44258027.19120625, 21174428.69540066],
            ],
        )
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[1]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [0.0, 19459910.91314028, 24053452.11581288],
                [48106904.23162583, 16648106.90423161, 22216035.63474388],
            ],
        )
        slice_ = Cube(CR.PETS_ARRAY_X_PETS, population=100000000).partitions[2]
        np.testing.assert_almost_equal(
            slice_.population_counts,
            [
                [21474773.60931439, 18272962.4838292, 0.0],
                [25808538.16300128, 23673997.41267791, 53751617.07632601],
            ],
        )
