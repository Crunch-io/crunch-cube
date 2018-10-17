'''Unit tests for the index functionality.'''

# pylint: disable=missing-docstring, invalid-name, redefined-outer-name

import pytest
from mock import Mock
import numpy as np

from cr.cube.cube_slice import CubeSlice


def test_index_performs_correct_division(index_fixture):
    cs, axis, base, expected = index_fixture
    actual = cs.index_table(axis, base)
    np.testing.assert_almost_equal(actual, expected)


@pytest.fixture(params=[
    (
        [[0.714285714286, 0.285714285714], [0.625, 0.375]],
        1, [0.6, 0.4],
        [[119.047619047619, 71.4285714285714], [104.16666666666667, 93.75]],
    ),
    (
        [[0.5, 0.4], [0.5, 0.6]],
        0, [0.6, 0.4],
        [[83.3333333333333, 66.6666666666667], [125, 150]],
    ),
])
def index_fixture(request):
    proportions, axis, base, expected = request.param
    cc = Mock()
    cc.ndim = 2
    cc.mr_dim_ind = None
    cc.proportions.return_value = np.array(proportions)
    cs = CubeSlice(cc, 0)
    base = np.array(base)
    return cs, axis, base, expected
