# encoding: utf-8

"""Second-order measure collection and the individual measures it composes."""

from __future__ import division

from cr.cube.util import lazyproperty


# === MEASURE COLLECTION ===


class StripeMeasures(object):
    """Intended to be a singleton for a given cube-result.

    It will give the same values if duplicated, just sacrificing some time and memory
    performance. Provides access to the variety of possible second-order measure objects
    for its cube-result. All construction and computation are lazy so only actually
    requested measures consume resources.
    """

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCounts measure object for this stripe."""
        raise NotImplementedError


# === INDIVIDUAL MEASURES ===


class _BaseSecondOrderMeasure(object):
    """Base class for all second-order measure objects."""

    @lazyproperty
    def blocks(self):
        """(base_values, subtotal_values) pair comprising the "blocks" of this measure.

        Use of this default implementation assumes implementation of a `._base_values`
        and `._subtotal_values` property in the subclass. A measure which is computed
        differently can override this `.blocks` property instead of implementing those
        two components.
        """
        raise NotImplementedError


class _UnweightedCounts(_BaseSecondOrderMeasure):
    """Provides the unweighted-counts measure for a stripe."""
