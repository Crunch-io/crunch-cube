# encoding: utf-8

import numpy as np


def assert_scale_means_equal(actual, expected):
    for act, exp in zip(actual, expected):
        if isinstance(exp, np.ndarray) and isinstance(act, np.ndarray):
            np.testing.assert_almost_equal(act, exp)
        elif isinstance(exp, list) and isinstance(act, list):
            assert_scale_means_equal(act, exp)
        else:
            assert act == exp
    assert True
