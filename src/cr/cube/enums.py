# encoding: utf-8

"""Enumerated sets related to cubes."""

from enum import Enum

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


class COLLATION_METHOD(Enum):
    """Enumerated values representing the methods of sorting dimension elements."""

    EXPLICIT_ORDER = "explicit"
    MARGINAL = "marginal"
    OPPOSING_ELEMENT = "opposing_element"
    OPPOSING_SUBTOTAL = "opposing_subtotal"
    PAYLOAD_ORDER = "payload_order"
