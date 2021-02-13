# encoding: utf-8

"""Integration-test suite for `cr.cube.stripe` sub-package."""

from cr.cube.cube import Cube
from cr.cube.stripe.assembler import StripeAssembler

from ..fixtures import CR


class DescribeStripeAssembler(object):
    """Integration-test suite for `cr.cube.stripe.assembler.StripeAssembler` object."""

    def it_provides_values_for_univariate_cat(self):
        cube = Cube(CR.UNIVARIATE_CATEGORICAL)
        assembler = StripeAssembler(cube, cube.dimensions[0], False, 0)

        assert assembler.unweighted_counts.tolist() == [10, 5]

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
