# encoding: utf-8

"""Unit test suite for `cr.cube.cube` module."""

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest

from cr.cube.cube import Cube


class DescribeBaseCrunchCube(object):
    def it_raises_a_type_error_if_not_serializable_object_provided(self):
        cube = Cube(None)
        with pytest.raises(TypeError) as pt_exc_info:
            cube._cube_dict
        exception = pt_exc_info.value
        assert (
            str(exception)
            == "Unsupported type <NoneType> provided. Cube response must be JSON (str) or dict."
        )
