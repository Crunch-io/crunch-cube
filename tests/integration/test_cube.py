# encoding: utf-8

"""Integration-test suite for `cr.cube.cube` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest

from cr.cube.cube import (
    Cube,
    _MeanMeasure,
    _Measures,
    _UnweightedCountMeasure,
    _WeightedCountMeasure,
)
from cr.cube.dimension import _ApparentDimensions, AllDimensions
from cr.cube.enums import DIMENSION_TYPE as DT

from ..fixtures import (
    CR,  # cube-response
    OL,  # overlaps
    NA,  # numeric-array
)


class DescribeIntegratedCube(object):
    """Integration-test suite for `cr.cube.cube.Cube` object."""

    def it_provides_values_for_cat_x_cat(self):
        cube = Cube(CR.CAT_X_CAT)

        assert cube.__repr__() == "Cube(name='v4', dimension_types='CAT x CAT')"
        assert cube.counts == pytest.approx(np.array([[5, 2], [5, 3]]))
        assert cube.counts_with_missings == pytest.approx(
            np.array([[5, 3, 2, 0], [5, 2, 3, 0], [0, 0, 0, 0]])
        )
        assert cube.cube_index == 0
        assert cube.description == "Pet Owners"
        assert cube.dimension_types == (DT.CAT, DT.CAT)
        assert isinstance(cube.dimensions, _ApparentDimensions)
        assert cube.has_weighted_counts is False
        assert cube.missing == 5
        assert cube.name == "v4"
        assert cube.ndim == 2
        assert cube.population_fraction == 1.0
        assert cube.title == "Pony Owners"
        assert cube.unweighted_counts == pytest.approx(np.array([[5, 2], [5, 3]]))
        assert cube.weighted_counts is None

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

        assert cube.counts == pytest.approx(np.array([[3, 2], [3, 1], [1, 1]]))

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

    def it_provides_access_to_the_overlaps_measure(self):
        cube_dict = OL.CAT_X_MR_SUB_X_MR_SEL
        measures = _Measures(
            cube_dict,
            AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
        )

        overlaps = measures.overlaps

        assert type(overlaps).__name__ == "_OverlapMeasure"

    def but_only_when_the_cube_response_contains_overlaps(self):
        cube_dict = CR.CAT_X_CAT
        measures = _Measures(cube_dict, None)

        overlaps = measures.overlaps

        assert overlaps is None

    def it_provides_access_to_the_mean_measure(self):
        cube_dict = CR.CAT_X_CAT_MEAN_WGTD
        measures = _Measures(
            cube_dict,
            AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
        )

        means = measures.means

        assert type(means).__name__ == "_MeanMeasure"

    def but_only_when_the_cube_response_contains_means(self):
        cube_dict = CR.CAT_X_CAT
        measures = _Measures(cube_dict, None)

        means = measures.means

        assert means is None

    def it_provides_the_means_missing_count_when_means_are_available(self):
        cube_dict = CR.CAT_X_CAT_MEAN_WGTD
        measures = _Measures(
            cube_dict,
            AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
        )
        missing_count = measures.missing_count
        assert missing_count == 3

    def it_provides_the_means_missing_count_when_sum_are_available(self):
        cube_dict = CR.SUM_CAT_X_MR
        measures = _Measures(
            cube_dict,
            AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
        )
        missing_count = measures.missing_count
        assert missing_count == 1

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
            (CR.CAT_X_CAT, "NoneType"),
        ),
    )
    def it_provides_access_to_wgtd_count_measure(self, cube_dict, expected_type_name):
        measures = _Measures(
            cube_dict,
            AllDimensions(dimension_dicts=cube_dict["result"]["dimensions"]),
        )

        weighted_counts = measures.weighted_counts

        assert type(weighted_counts).__name__ == expected_type_name


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
