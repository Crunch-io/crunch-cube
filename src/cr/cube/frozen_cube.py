# encoding: utf-8

"""Provides the FrozenCube class.

FrozenCube is the main API class for manipulating Crunch.io JSON cube responses.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import json

import numpy as np

from cr.cube.cubepart import CubePartition
from cr.cube.dimension import AllDimensions
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty

np.seterr(divide="ignore", invalid="ignore")


class FrozenCube(object):
    """Provides access to individual slices on a cube-result.

    It also provides some attributes of the overall cube-result.
    """

    def __init__(
        self,
        response,
        transforms=None,
        first_cube_of_tab=False,
        population=None,
        mask_size=0,
    ):
        self._cube_response_arg = response
        self._transforms_dict = {} if transforms is None else transforms
        self._first_cube_of_tab = first_cube_of_tab
        self._population = 0 if population is None else population
        self._mask_size = mask_size

    def __repr__(self):
        """Provide text representation suitable for working at console.

        Falls back to a default repr on exception, such as might occur in
        unit tests where object need not otherwise be provided with all
        instance variable values.
        """
        try:
            dimensionality = " x ".join(dt.name for dt in self.dimension_types)
            return "%s(name='%s', dimension_types='%s')" % (
                type(self).__name__,
                self.name,
                dimensionality,
            )
        except Exception:
            return super(FrozenCube, self).__repr__()

    @lazyproperty
    def partitions(self):
        """Sequence of _Slice, _Strand, or _Nub objects from this cube-result."""
        return tuple(
            CubePartition.factory(
                self,
                slice_idx=slice_idx,
                transforms=self._transforms_dict,
                population=self._population,
                ca_as_0th=self._ca_as_0th,
                mask_size=self._mask_size,
            )
            for slice_idx in self._slice_idxs
        )

    @lazyproperty
    def base_counts(self):
        return self._measures.unweighted_counts.raw_cube_array[self._valid_idxs]

    @lazyproperty
    def counts(self):
        return self.counts_with_missings[self._valid_idxs]

    @lazyproperty
    def counts_with_missings(self):
        return self._measure(self.is_weighted).raw_cube_array

    @lazyproperty
    def description(self):
        """Return the description of the cube."""
        if not self.dimensions:
            return None
        return self.dimensions[0].description

    @lazyproperty
    def dimension_types(self):
        """Tuple of DIMENSION_TYPE member for each dimension of cube."""
        return tuple(d.dimension_type for d in self.dimensions)

    @lazyproperty
    def dimensions(self):
        """_ApparentDimension object providing access to visible dimensions.

        A cube involving a multiple-response (MR) variable has two dimensions
        for that variable (subvariables and categories dimensions), but is
        "collapsed" into a single effective dimension for cube-user purposes
        (its categories dimension is supressed). This collection will contain
        a single dimension for each MR variable and therefore may have fewer
        dimensions than appear in the cube response.
        """
        return self._all_dimensions.apparent_dimensions

    @lazyproperty
    def has_means(self):
        """True if cube includes a means measure."""
        return self._measures.means is not None

    @lazyproperty
    def is_weighted(self):
        """True if cube response contains weighted data."""
        return self._measures.is_weighted

    @lazyproperty
    def missing(self):
        """Get missing count of a cube."""
        return self._measures.missing_count

    @lazyproperty
    def name(self):
        """Return the name of the cube.

        If the cube has 2 diensions, return the name of the second one. In case
        of a different number of dimensions, default to returning the name of
        the last one. In case of no dimensions, return the empty string.
        """
        if not self.dimensions:
            return None
        return self.dimensions[0].name

    @lazyproperty
    def ndim(self):
        """int count of dimensions for this cube."""
        return len(self.dimensions)

    @lazyproperty
    def population_fraction(self):
        """The filtered/unfiltered ratio for cube response.

        This value is required for properly calculating population on a cube
        where a filter has been applied. Returns 1.0 for an unfiltered cube.
        Returns `np.nan` if the unfiltered count is zero, which would
        otherwise result in a divide-by-zero error.
        """
        return self._measures.population_fraction

    @lazyproperty
    def _all_dimensions(self):
        """The AllDimensions object for this cube.

        The AllDimensions object provides access to all the dimensions appearing in the
        cube response, not only apparent dimensions (those that appear to a user). It
        also provides access to an _ApparentDimensions object which contains only those
        user-apparent dimensions (basically the categories dimension of each MR
        dimension-pair is suppressed).
        """
        return AllDimensions(dimension_dicts=self._cube_dict["result"]["dimensions"])

    @lazyproperty
    def _ca_as_0th(self):
        """True if slicing is to be performed in so-called "CA-as-0th" mode.

        In this mode, a categorical-array (CA) cube (2D) is sliced into a sequence of 1D
        slices, each of which represents one subvariable of the CA variable. Normally,
        a 2D cube-result becomes a single slice.
        """
        return self._first_cube_of_tab and self.dimension_types[0] == DT.CA

    @lazyproperty
    def _cube_dict(self):
        """dict containing raw cube response, parsed from JSON payload."""
        try:
            cube_response = self._cube_response_arg
            # ---parse JSON to a dict when constructed with JSON---
            cube_dict = (
                cube_response
                if isinstance(cube_response, dict)
                else json.loads(cube_response)
            )
            # ---cube is 'value' item in a shoji response---
            return cube_dict.get("value", cube_dict)
        except TypeError:
            raise TypeError(
                "Unsupported type <%s> provided. Cube response must be JSON "
                "(str) or dict." % type(self._cube_response_arg).__name__
            )

    def _measure(self, weighted):
        """_BaseMeasure subclass representing primary measure for this cube.

        If the cube response includes a means measure, the return value is
        means. Otherwise it is counts, with the choice between weighted or
        unweighted determined by *weighted*.

        Note that weighted counts are provided on an "as-available" basis.
        When *weighted* is True and the cube response is not weighted,
        unweighted counts are returned.
        """
        return (
            self._measures.means
            if self._measures.means is not None
            else self._measures.weighted_counts
            if weighted
            else self._measures.unweighted_counts
        )

    @lazyproperty
    def _measures(self):
        """_Measures object for this cube.

        Provides access to unweighted counts, and weighted counts and/or means
        when available.
        """
        return _Measures(self._cube_dict, self._all_dimensions)

    @lazyproperty
    def _slice_idxs(self):
        """Iterable of contiguous int indicies for slices to be produced.

        This value is to help cube-section construction which does not by itself know
        how many slices are in a cube-result.
        """
        if self.ndim < 3 and not self._ca_as_0th:
            return (0,)
        return range(len(self.dimensions[0].valid_elements))

    @lazyproperty
    def _valid_idxs(self):
        return np.ix_(
            *tuple(d.valid_elements.element_idxs for d in self._all_dimensions)
        )


class _Measures(object):
    """Provides access to measures contained in cube response."""

    def __init__(self, cube_dict, all_dimensions):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions

    @lazyproperty
    def is_weighted(self):
        """True if weights have been applied to the measure(s) for this cube.

        Unweighted counts are available for all cubes. Weighting applies to
        any other measures provided by the cube.
        """
        cube_dict = self._cube_dict
        if cube_dict.get("query", {}).get("weight") is not None:
            return True
        if cube_dict.get("weight_var") is not None:
            return True
        if cube_dict.get("weight_url") is not None:
            return True
        unweighted_counts = cube_dict["result"]["counts"]
        count_data = cube_dict["result"]["measures"].get("count", {}).get("data")
        if unweighted_counts != count_data:
            return True
        return False

    @lazyproperty
    def means(self):
        """_MeanMeasure object providing access to means values.

        None when the cube response does not contain a mean measure.
        """
        mean_measure_dict = (
            self._cube_dict.get("result", {}).get("measures", {}).get("mean")
        )
        if mean_measure_dict is None:
            return None
        return _MeanMeasure(self._cube_dict, self._all_dimensions)

    @lazyproperty
    def missing_count(self):
        """numeric representing count of missing rows in cube response."""
        if self.means:
            return self.means.missing_count
        return self._cube_dict["result"].get("missing", 0)

    @lazyproperty
    def population_fraction(self):
        """The filtered/unfiltered ratio for cube response.

        This value is required for properly calculating population on a cube
        where a filter has been applied. Returns 1.0 for an unfiltered cube.
        Returns `np.nan` if the unfiltered count is zero, which would
        otherwise result in a divide-by-zero error.
        """
        numerator = self._cube_dict["result"].get("filtered", {}).get("weighted_n")
        denominator = self._cube_dict["result"].get("unfiltered", {}).get("weighted_n")
        try:
            return numerator / denominator
        except ZeroDivisionError:
            return np.nan
        except Exception:
            return 1.0

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCountMeasure object for this cube.

        This object provides access to unweighted counts for this cube,
        whether or not the cube contains weighted counts.
        """
        return _UnweightedCountMeasure(self._cube_dict, self._all_dimensions)

    @lazyproperty
    def weighted_counts(self):
        """_WeightedCountMeasure object for this cube.

        This object provides access to weighted counts for this cube, if
        available. If the cube response is not weighted, the
        _UnweightedCountMeasure object for this cube is returned.
        """
        return _WeightedCountMeasure(self._cube_dict, self._all_dimensions)


class _BaseMeasure(object):
    """Base class for measure objects."""

    def __init__(self, cube_dict, all_dimensions):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions

    @lazyproperty
    def raw_cube_array(self):
        """Return read-only ndarray of measure values from cube-response.

        The shape of the ndarray mirrors the shape of the (raw) cube
        response. Specifically, it includes values for missing elements, any
        MR_CAT dimensions, and any prunable rows and columns.
        """
        array = np.array(self._flat_values).reshape(self._all_dimensions.shape)
        # ---must be read-only to avoid hard-to-find bugs---
        array.flags.writeable = False
        return array

    @lazyproperty
    def _flat_values(self):
        """Return tuple of mean values as found in cube response.

        This property must be implemented by each subclass.
        """
        raise NotImplementedError("must be implemented by each subclass")


class _MeanMeasure(_BaseMeasure):
    """Statistical mean values from a cube-response."""

    @lazyproperty
    def missing_count(self):
        """numeric representing count of missing rows reflected in response."""
        return self._cube_dict["result"]["measures"]["mean"].get("n_missing", 0)

    @lazyproperty
    def _flat_values(self):
        """Return tuple of mean values as found in cube response.

        Mean data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        return tuple(
            np.nan if type(x) is dict else x
            for x in self._cube_dict["result"]["measures"]["mean"]["data"]
        )


class _UnweightedCountMeasure(_BaseMeasure):
    """Unweighted counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """tuple of int counts before weighting."""
        return tuple(self._cube_dict["result"]["counts"])


class _WeightedCountMeasure(_BaseMeasure):
    """Weighted counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """tuple of numeric counts after weighting."""
        return tuple(self._cube_dict["result"]["measures"]["count"]["data"])
