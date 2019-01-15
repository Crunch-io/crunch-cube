# encoding: utf-8

"""Unit test suite for cr.cube.cube_slice module."""

from __future__ import absolute_import, division, print_function, unicode_literals

from cr.cube.enum import _DimensionType


class Describe_DimensionType(object):
    def it_provides_a_console_friendly_representation(self):
        dimension_type = _DimensionType("ALTERNATE_UNIVERSE")
        repr_ = repr(dimension_type)
        assert repr_ == "<DIMENSION_TYPE.ALTERNATE_UNIVERSE>"

    def it_provides_a_user_friendly_str_representation(self):
        dimension_type = _DimensionType("QUANTUM_MANIFOLD")
        str_ = str(dimension_type)
        assert str_ == "DIMENSION_TYPE.QUANTUM_MANIFOLD"

    def it_knows_its_name(self):
        dimension_type = _DimensionType("WORM_HOLE")
        name = dimension_type.name
        assert name == "WORM_HOLE"
