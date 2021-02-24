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

        assert assembler.unweighted_counts.tolist() == [10, 5]
        assert assembler.weighted_counts.tolist() == [10, 5]

    @pytest.mark.xfail(reason="WIP", raises=NotImplementedError, strict=True)
    def it_provides_values_for_univariate_mr(self):
        cube = Cube(CR.MR_WGTD)
        assembler = StripeAssembler(cube, cube.dimensions[0], False, 0)

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
