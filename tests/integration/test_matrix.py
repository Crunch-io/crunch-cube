# encoding: utf-8

"""Integration-test suite for `cr.cube.cubepart` module."""

import numpy as np
import pytest

from cr.cube.cube import Cube
from cr.cube.cubepart import _Slice

from ..fixtures import CR


class DescribeAssembler(object):
    """Integration-test suite for `cr.cube.matrix.Assembler`."""

    @pytest.mark.xfail(reason="WIP", raises=NotImplementedError, strict=True)
    def it_computes_assembled_ucounts_for_cat_hs_x_cat_hs_hiddens(self):
        """Assembler inserts, hides, prunes, and places in payload order."""
        slice_ = _Slice(
            Cube(CR.CAT_HS_X_CAT_HS_EMPTIES),
            slice_idx=0,
            transforms={
                "rows_dimension": {"elements": {"2": {"hide": True}}, "prune": True},
                "columns_dimension": {"elements": {"2": {"hide": True}}, "prune": True},
            },
            population=None,
            mask_size=0,
        )

        unweighted_counts = slice_._assembler.unweighted_counts

        assert np.array_equal(
            unweighted_counts,
            [
                [118, 65, 53, 33, 33],
                [40, 32, 8, 12, 12],
                [168, 66, 102, 82, 82],
                [163, 93, 70, 190, 190],
                [331, 159, 172, 272, 272],
                [100, 49, 51, 23, 23],
                [341, 175, 166, 234, 234],
            ],
        )
