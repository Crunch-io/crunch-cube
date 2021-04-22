# encoding: utf-8

"""Enumerated sets related to cubes."""

import enum

from cr.cube.util import lazyproperty


class _DimensionType(object):
    """Member of the DIMENSION_TYPE enumeration."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return "<DIMENSION_TYPE.%s>" % self._name

    def __str__(self):
        return "DIMENSION_TYPE.%s" % self._name

    @lazyproperty
    def name(self):
        """str like 'CA_CAT' for this dimension type."""
        return self._name


class DIMENSION_TYPE(object):
    """Enumerated values representing the various types of dimension."""

    # ---member definitions---
    BINNED_NUMERIC = _DimensionType("BINNED_NUMERIC")
    CAT = _DimensionType("CAT")
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
    MARGINAL = "marginal"
    OPPOSING_ELEMENT = "opposing_element"
    OPPOSING_SUBTOTAL = "opposing_subtotal"
    PAYLOAD_ORDER = "payload_order"
    UNIVARIATE_MEASURE = "univariate_measure"


class MEASURE(enum.Enum):
    """Enumerated values representing the second-order measures."""

    # --- value for each member should match the export measure keyname ---
    COL_INDEX = "col_index"
    COL_PERCENT = "col_percent"
    COL_SHARE_SUM = "col_share_sum"
    MEAN = "mean"
    ROW_SHARE_SUM = "row_share_sum"
    SUM = "sum"
    TABLE_STDERR = "table_stderr"
    TOTAL_SHARE_SUM = "total_share_sum"
    UNWEIGHTED_COUNT = "count_unweighted"
    WEIGHTED_COUNT = "count_weighted"
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
    VALID_COUNT_UNWEIGHTED = "valid_count_unweighted"
    VALID_COUNT_WEIGHTED = "valid_count_weighted"


NUMERIC_MEASURES = frozenset(
    (
        CUBE_MEASURE.SUM,
        CUBE_MEASURE.MEAN,
        CUBE_MEASURE.STDDEV,
        CUBE_MEASURE.VALID_COUNT_UNWEIGHTED,
        CUBE_MEASURE.VALID_COUNT_WEIGHTED,
        MEASURE.COL_SHARE_SUM,
        MEASURE.ROW_SHARE_SUM,
        MEASURE.TOTAL_SHARE_SUM,
    )
)
