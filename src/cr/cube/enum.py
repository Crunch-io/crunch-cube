# encoding: utf-8

"""Enumerated sets related to cubes."""


class _DimensionType(object):
    """Member of the DIMENSION_TYPE enumeration."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return '<DIMENSION_TYPE.%s>' % self._name

    def __str__(self):
        return 'DIMENSION_TYPE.%s' % self._name


class DIMENSION_TYPE(object):
    """Enumerated values representing the various types of dimension."""

    # ---member definitions---
    BINNED_NUMERIC = _DimensionType('BINNED_NUMERIC')
    CATEGORICAL = _DimensionType('CATEGORICAL')
    CATEGORICAL_ARRAY = _DimensionType('CATEGORICAL_ARRAY')
    CA_CAT = _DimensionType('CA_CAT')
    DATETIME = _DimensionType('DATETIME')
    LOGICAL = _DimensionType('LOGICAL')
    MR_CAT = _DimensionType('MR_CAT')
    MULTIPLE_RESPONSE = _DimensionType('MULTIPLE_RESPONSE')
    TEXT = _DimensionType('TEXT')

    # ---aliases---
    CA = CATEGORICAL_ARRAY
    CAT = CATEGORICAL
    CAT_ARRAY = CATEGORICAL_ARRAY
    CA_SUBVAR = CATEGORICAL_ARRAY
    MR = MULTIPLE_RESPONSE
    MR_SELECTIONS = MR_CAT
    MR_SUBVAR = MULTIPLE_RESPONSE

    # ---subsets---
    ARRAY_TYPES = frozenset((CATEGORICAL_ARRAY, MULTIPLE_RESPONSE))
