# encoding: utf-8

"""Enumerated sets related to cubes."""

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
    ARRAY_TYPES = frozenset((CA_SUBVAR, MR_SUBVAR))

    # ---allowed types for pairwise comparison---
    ALLOWED_PAIRWISE_TYPES = frozenset((CAT, CA_CAT, BINNED_NUMERIC, DATETIME, TEXT))
