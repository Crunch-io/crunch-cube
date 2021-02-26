# encoding: utf-8

"""Integration-test suite for `cr.cube.stripe` sub-package."""

import pytest

from cr.cube.cube import Cube
from cr.cube.stripe.assembler import StripeAssembler

from ..fixtures import CR


class DescribeStripeAssembler(object):
    """Integration-test suite for `cr.cube.stripe.assembler.StripeAssembler` object."""

    def it_provides_values_for_univariate_cat(self):
        cube = Cube(CR.UNIVARIATE_CATEGORICAL)
        assembler = StripeAssembler(cube, cube.dimensions[0], False, 0)

        assert assembler.table_proportion_stddevs == pytest.approx(
            [0.4714045, 0.4714045]
        )
        assert assembler.table_proportion_stderrs == pytest.approx(
            [0.1217161, 0.1217161]
        )
        assert assembler.table_proportions == pytest.approx([0.6666667, 0.3333333])
        assert assembler.unweighted_bases.tolist() == [15, 15]
        assert assembler.unweighted_counts.tolist() == [10, 5]
        assert assembler.weighted_bases.tolist() == [15, 15]
        assert assembler.weighted_counts.tolist() == [10, 5]

    def it_provides_values_for_univariate_mr(self):
        cube = Cube(CR.MR_WGTD)
        assembler = StripeAssembler(cube, cube.dimensions[0], False, 0)

        assert assembler.table_proportion_stddevs == pytest.approx(
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
        assert assembler.table_proportion_stderrs == pytest.approx(
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
        assert assembler.table_proportions == pytest.approx(
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
        assert assembler.unweighted_bases.tolist() == [
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
        assert assembler.unweighted_counts.tolist() == [
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
        assert assembler.weighted_bases == pytest.approx(
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
        assert assembler.weighted_counts == pytest.approx(
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
