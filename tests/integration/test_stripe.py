# encoding: utf-8

"""Integration-test suite for `cr.cube.stripe` sub-package."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Strand

from ..fixtures import CR


class TestStripeAssembler:
    """Integration-test suite for `cr.cube.stripe.assembler.StripeAssembler` object."""

    def test_it_provides_values_for_univariate_cat(self):
        cube = Cube(CR.UNIVARIATE_CATEGORICAL)
        strand = _Strand(cube, None, None, False, 0, None)

        assert strand.inserted_row_idxs == ()
        assert strand.row_count == 2
        assert strand.row_labels.tolist() == ["C", "E"]
        assert strand.rows_dimension_fills == (None, None)
        assert strand.scale_mean == pytest.approx(1.666667)
        assert strand.scale_median == pytest.approx(1.0)
        assert strand.scale_stddev == pytest.approx(0.9428090)
        assert strand.scale_stderr == pytest.approx(0.2434322)
        assert strand.table_base_range == pytest.approx([15, 15])
        assert strand.table_margin_range == pytest.approx([15, 15])
        assert strand.table_proportion_stddevs == pytest.approx([0.4714045, 0.4714045])
        assert strand.table_proportion_stderrs == pytest.approx([0.1217161, 0.1217161])
        assert strand.table_proportions == pytest.approx([0.6666667, 0.3333333])
        assert strand.unweighted_bases.tolist() == [15, 15]
        assert strand.unweighted_counts.tolist() == [10, 5]
        assert strand.weighted_bases.tolist() == [15, 15]
        assert strand.weighted_counts.tolist() == [10, 5]

    def test_it_provides_values_for_univariate_cat_means(self):
        cube = Cube(CR.CAT_MEANS_HS)
        strand = _Strand(cube, None, None, False, 0, None)

        assert strand.inserted_row_idxs == (3, 4)
        assert strand.means == pytest.approx(
            [19.85556, 13.85417, 52.7894736842, np.nan, np.nan], nan_ok=True
        )
        assert strand.row_count == 5
        assert strand.row_labels.tolist() == [
            "Yes",
            "No",
            "I'm not sur",
            "Seen the Ad",
            "Not Seen th",
        ]
        assert strand.rows_dimension_fills == (None, None, None, None, None)
        assert strand.scale_mean is None
        assert strand.scale_median is None
        assert strand.scale_stddev is None
        assert strand.scale_stderr is None
        assert strand.table_base_range == pytest.approx([661, 661])
        # for a cube with numeric measure like mean, table margin and table base are the
        # same because they are both calculated on the (unweighted) valid-counts.
        assert strand.table_margin_range == pytest.approx([661, 661])

    def test_it_provides_values_for_univariate_mr(self):
        cube = Cube(CR.MR_WGTD)
        strand = _Strand(cube, None, None, False, 0, None)

        assert strand.inserted_row_idxs == ()
        assert strand.row_count == 9
        assert strand.row_labels.tolist() == [
            "liver",
            "thalmus",
            "heart",
            "tripe",
            "kidney",
            "lungs",
            "other",
            "Don't know",
            "None of the",
        ]
        assert strand.rows_dimension_fills == (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        assert strand.scale_mean is None
        assert strand.scale_median is None
        assert strand.scale_stddev is None
        assert strand.scale_stderr is None
        assert strand.table_base_range == pytest.approx([43504, 43504])
        assert strand.table_margin_range == pytest.approx([43009.56, 43009.56])
        assert strand.table_proportion_stddevs == pytest.approx(
            [
                0.4986677,
                0.4036694,
                0.4486895,
                0.3885031,
                0.4918164,
                0.3241297,
                0.1177077,
                0.1979895,
                0.2601507,
            ]
        )
        assert strand.table_proportion_stderrs == pytest.approx(
            [
                0.0024045217,
                0.0019464504,
                0.0021635322,
                0.0018733202,
                0.0023714856,
                0.0015629185,
                0.0005675739,
                0.0009546841,
                0.0012544185,
            ]
        )
        assert strand.table_proportions == pytest.approx(
            [
                0.4635233,
                0.2049559,
                0.2793696,
                0.1852536,
                0.5900924,
                0.1192902,
                0.01405258,
                0.04087024,
                0.07300864,
            ]
        )
        assert strand.unweighted_bases.tolist() == [
            43504,
            43504,
            43504,
            43504,
            43504,
            43504,
            43504,
            43504,
            43504,
        ]
        assert strand.unweighted_counts.tolist() == [
            21545,
            9256,
            13412,
            8562,
            27380,
            5041,
            676,
            1281,
            3112,
        ]
        assert strand.weighted_bases == pytest.approx(
            [
                43009.56,
                43009.56,
                43009.56,
                43009.56,
                43009.56,
                43009.56,
                43009.56,
                43009.56,
                43009.56,
            ]
        )
        assert strand.weighted_counts == pytest.approx(
            [
                19935.93,
                8815.065,
                12015.56,
                7967.675,
                25379.62,
                5130.620,
                604.3953,
                1757.811,
                3140.070,
            ]
        )

    def test_it_provides_values_for_univariate_mr_means(self):
        cube = Cube(CR.MR_MEAN_FILT_WGTD)
        strand = _Strand(cube, None, None, False, 0, None)

        assert strand.means == pytest.approx(
            [3.72405146, 2.57842929, 2.21859327, 1.86533494]
        )

    @pytest.mark.parametrize(
        "fixture, measure, direction, expected_value",
        (
            (CR.CAT, "base_unweighted", "ascending", [0, 1, 2, 3, 4]),
            (CR.CAT, "base_unweighted", "descending", [4, 3, 2, 1, 0]),
            (CR.ECON_BLAME_WITH_HS, "base_weighted", "ascending", [0, 1, 2, 3, 4, -1]),
            (CR.ECON_BLAME_WITH_HS, "base_weighted", "descending", [-1, 4, 3, 2, 1, 0]),
            (CR.CAT_HS_MT, "count_unweighted", "ascending", [3, 4, 2, 0, 1, -1, -2]),
            (CR.CAT_HS_MT, "count_unweighted", "descending", [-2, -1, 1, 0, 2, 4, 3]),
            (CR.MR_WGTD, "count_weighted", "ascending", [6, 7, 8, 5, 3, 1, 2, 0, 4]),
            (CR.MR_WGTD, "count_weighted", "descending", [4, 0, 2, 1, 3, 5, 8, 7, 6]),
            (CR.CAT_MEANS_HS, "mean", "ascending", [1, 0, 2, -2, -1]),
            (CR.CAT_MEANS_HS, "mean", "descending", [-2, -1, 2, 0, 1]),
            (CR.CAT, "percent", "ascending", [4, 3, 2, 0, 1]),
            (CR.CAT, "percent", "descending", [1, 0, 2, 3, 4]),
            (CR.ECON_BLAME_WITH_HS, "percent_moe", "ascending", [3, 4, 2, 0, 1, -1]),
            (CR.ECON_BLAME_WITH_HS, "percent_moe", "descending", [-1, 1, 0, 2, 4, 3]),
            (CR.CAT_HS_MT, "percent_stddev", "ascending", [3, 4, 2, 0, 1, -1, -2]),
            (CR.CAT_HS_MT, "percent_stddev", "descending", [-2, -1, 1, 0, 2, 4, 3]),
            (CR.ECON_BLAME_WITH_HS, "percent_stderr", "ascending", [3, 4, 2, 0, 1, -1]),
            (CR.MR_WGTD, "percent_stderr", "descending", [0, 4, 2, 1, 3, 5, 8, 7, 6]),
            (CR.MR_WGTD, "percent_stderr", "ascending", [6, 7, 8, 5, 3, 1, 2, 4, 0]),
            (CR.CAT_MEAN, "sum", "ascending", [1, 3, 0, 2, 4]),
            (CR.CAT_MEAN, "sum", "descending", [4, 2, 0, 3, 1]),
        ),
    )
    def test_it_computes_the_sort_by_measure_value_row_order_to_help(
        self, fixture, measure, direction, expected_value
    ):
        transforms = {
            "rows_dimension": {
                "order": {
                    "type": "univariate_measure",
                    "measure": measure,
                    "direction": direction,
                }
            }
        }
        cube = Cube(fixture, transforms=transforms)
        stripe = _Strand(cube, transforms, None, False, 0, None)

        assert stripe.row_order().tolist() == expected_value
