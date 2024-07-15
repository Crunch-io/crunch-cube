# encoding: utf-8

"""Unit test suite for cr.cube.enums module."""

from cr.cube.enums import (
    CUBE_MEASURE,
    MEASURE,
    NUMERIC_CUBE_MEASURES,
    NUMERIC_MEASURES,
    _DimensionType,
)


class Describe_DimensionType:
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


class TestCubeMeasures:
    def test_numeric_cube_measures_intersection(self):
        intersection = NUMERIC_CUBE_MEASURES & {m for m in CUBE_MEASURE}
        expected_intersection = {
            MEASURE.MEAN,
            MEASURE.MEDIAN,
            MEASURE.SUM,
            MEASURE.STDDEV,
            MEASURE.UNWEIGHTED_VALID_COUNT,
            MEASURE.WEIGHTED_VALID_COUNT,
        }
        assert sorted(list([m.value for m in intersection])) == sorted(
            list([m.value for m in expected_intersection])
        )

    def test_numeric_cube_measures_difference(self):
        difference = {m.value for m in NUMERIC_MEASURES} - {
            m.value for m in NUMERIC_CUBE_MEASURES
        }
        expected_difference = {
            "smoothed_mean",
            "total_share_sum",
            "row_share_sum",
            "col_share_sum",
        }
        assert list(sorted(difference)) == list(sorted(expected_difference))
