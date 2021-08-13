# encoding: utf-8

"""Enumerated sets related to cubes."""

import enum

from cr.cube.util import lazyproperty


class _DimensionType:
    """Member of the DIMENSION_TYPE enumeration."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<DIMENSION_TYPE.{self._name}>"

    def __str__(self):
        return f"DIMENSION_TYPE.{self._name}"

    @lazyproperty
    def name(self):
        """str like 'CA_CAT' for this dimension type."""
        return self._name


class DIMENSION_TYPE:
    """Enumerated values representing the various types of dimension."""

    # ---member definitions---
    BINNED_NUMERIC = _DimensionType("BINNED_NUMERIC")
    CAT = _DimensionType("CAT")
    CAT_DATE = _DimensionType("CAT_DATE")
    CA_CAT = _DimensionType("CA_CAT")
    CA_SUBVAR = _DimensionType("CA_SUBVAR")
    DATETIME = _DimensionType("DATETIME")
    LOGICAL = _DimensionType("LOGICAL")
    MR_CAT = _DimensionType("MR_CAT")
    MR_SUBVAR = _DimensionType("MR_SUBVAR")
    NUM_ARRAY = _DimensionType("NUM_ARRAY")
    TEXT = _DimensionType("TEXT")

    # ---aliases---
    CA = CA_SUBVAR
    CATEGORICAL = CAT
    CATEGORICAL_ARRAY = CA_SUBVAR
    CAT_ARRAY = CA_SUBVAR
    MR = MR_SUBVAR
    MR_SELECTIONS = MR_CAT
    MULTIPLE_RESPONSE = MR_SUBVAR

    # ---subsets---
    ARRAY_TYPES = frozenset((CA_SUBVAR, MR_SUBVAR, NUM_ARRAY))

    # ---allowed types for pairwise comparison---
    ALLOWED_PAIRWISE_TYPES = frozenset(
        (BINNED_NUMERIC, CA, CAT, CA_CAT, DATETIME, MR, TEXT)
    )


class COLLATION_METHOD(enum.Enum):
    """Enumerated values representing the methods of sorting dimension elements."""

    EXPLICIT_ORDER = "explicit"
    LABEL = "label"
    MARGINAL = "marginal"
    OPPOSING_ELEMENT = "opposing_element"
    OPPOSING_INSERTION = "opposing_insertion"
    PAYLOAD_ORDER = "payload_order"
    UNIVARIATE_MEASURE = "univariate_measure"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class MARGINAL(enum.Enum):
    """Enumerated values representing the (second-order) marginals."""

    BASE = "unweighted_base"
    MARGIN = "weighted_base"
    MARGIN_PROPORTION = "table_proportion"
    SCALE_MEAN = "scale_mean"
    SCALE_MEAN_STDDEV = "scale_mean_stddev"
    SCALE_MEAN_STDERR = "scale_mean_stderr"
    SCALE_MEDIAN = "scale_median"


class MARGINAL_ORIENTATION(enum.Enum):
    """Enumerated values representing orientation of a marginal."""

    ROWS = "rows"
    COLUMNS = "columns"


class MEASURE(enum.Enum):
    """Enumerated values representing the second-order measures."""

    # --- value for each member should match the export measure keyname ---
    COLUMN_BASE_UNWEIGHTED = "col_base_unweighted"
    COLUMN_BASE_WEIGHTED = "col_base_weighted"
    COLUMN_INDEX = "col_index"
    COLUMN_PERCENT = "col_percent"
    COLUMN_PERCENT_MOE = "col_percent_moe"
    COLUMN_SHARE_SUM = "col_share_sum"
    COLUMN_STDDEV = "col_std_dev"
    COLUMN_STDERR = "col_std_err"
    MEAN = "mean"
    PAIRWISE_T_TEST = "pairwise_t_test"
    POPULATION = "population"
    POPULATION_MOE = "population_moe"
    PVALUES = "p_value"
    ROW_BASE_UNWEIGHTED = "row_base_unweighted"
    ROW_BASE_WEIGHTED = "row_base_weighted"
    ROW_PERCENT = "row_percent"
    ROW_PERCENT_MOE = "row_percent_moe"
    ROW_SHARE_SUM = "row_share_sum"
    ROW_STDDEV = "row_std_dev"
    ROW_STDERR = "row_std_err"
    SMOOTHED_MEAN = "smoothed_mean"
    SMOOTHED_COL_PERCENT = "smoothed_col_percent"
    SMOOTHED_COL_INDEX = "smoothed_col_index"
    STDDEV = "stddev"
    SUM = "sum"
    TABLE_BASE_UNWEIGHTED = "table_base_unweighted"
    TABLE_BASE_WEIGHTED = "table_base_weighted"
    TABLE_PERCENT = "table_percent"
    TABLE_PERCENT_MOE = "table_percent_moe"
    TABLE_STDDEV = "table_std_dev"
    TABLE_STDERR = "table_std_err"
    TOTAL_SHARE_SUM = "total_share_sum"
    UNWEIGHTED_COUNT = "count_unweighted"
    UNWEIGHTED_VALID_COUNT = "valid_count_unweighted"
    WEIGHTED_COUNT = "count_weighted"
    WEIGHTED_VALID_COUNT = "valid_count_weighted"
    Z_SCORE = "z_score"


class CUBE_MEASURE(enum.Enum):
    """Enumerated values representing cube measures."""

    COVARIANCE = "covariance"
    COUNT = "count"
    MEAN = "mean"
    OVERLAP = "overlap"
    STDDEV = "stddev"
    SUM = "sum"
    VALID_OVERLAP = "valid_overlap"
    UNWEIGHTED_VALID_COUNT = "valid_count_unweighted"
    WEIGHTED_VALID_COUNT = "valid_count_weighted"


NUMERIC_CUBE_MEASURES = frozenset(
    (
        CUBE_MEASURE.MEAN,
        CUBE_MEASURE.SUM,
        CUBE_MEASURE.STDDEV,
        CUBE_MEASURE.UNWEIGHTED_VALID_COUNT,
        CUBE_MEASURE.WEIGHTED_VALID_COUNT,
    )
)

NUMERIC_MEASURES = frozenset(
    (
        MEASURE.MEAN,
        MEASURE.SUM,
        MEASURE.STDDEV,
        MEASURE.WEIGHTED_VALID_COUNT,
        MEASURE.UNWEIGHTED_VALID_COUNT,
        MEASURE.SMOOTHED_MEAN,
        MEASURE.COLUMN_SHARE_SUM,
        MEASURE.ROW_SHARE_SUM,
        MEASURE.TOTAL_SHARE_SUM,
    )
)
