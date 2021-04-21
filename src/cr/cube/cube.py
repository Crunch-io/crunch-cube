# encoding: utf-8

"""Provides the CubeSet and Cube classes.

CubeSet is the main API class for manipulating Crunch.io JSON cube responses.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json

import numpy as np

from cr.cube.cubepart import CubePartition
from cr.cube.dimension import AllDimensions
from cr.cube.enums import CUBE_MEASURE, DIMENSION_TYPE as DT, NUMERIC_MEASURES
from cr.cube.util import lazyproperty

np.seterr(divide="ignore", invalid="ignore")


class CubeSet(object):
    """Represents a multi-cube cube-response.

    Also works just fine for a single cube-response passed inside a sequence, allowing
    uniform handling of single and multi-cube responses.

    `cube_responses` is a sequence of cube-response dicts received from Crunch. The
    sequence can contain a single item, such as a cube-response for a slide, but it must
    be contained in a sequence. A tabbook cube-response sequence can be passed as it was
    received.

    `transforms` is a sequence of transforms dicts corresponding in order to the
    cube-responses. `population` is the estimated target population and is used when
    a population-projection measure is requested. `min_base` is an integer representing
    the minimum sample-size used for indicating values that are unreliable by reason of
    insufficient sample (base).
    """

    def __init__(self, cube_responses, transforms, population, min_base):
        self._cube_responses = cube_responses
        self._transforms_dicts = transforms
        self._population = population
        self._min_base = min_base

    @lazyproperty
    def available_measures(self):
        """frozenset of available measures of the first cube in this set."""
        return self._cubes[0].available_measures

    @lazyproperty
    def can_show_pairwise(self):
        """True if all 2D cubes in a multi-cube set can provide pairwise comparison."""
        if len(self._cubes) < 2:
            return False

        return all(
            all(dt in DT.ALLOWED_PAIRWISE_TYPES for dt in cube.dimension_types[-2:])
            and cube.ndim >= 2
            for cube in self._cubes[1:]
        )

    @lazyproperty
    def description(self):
        """str description of first cube in this set."""
        return self._cubes[0].description

    @lazyproperty
    def has_weighted_counts(self):
        """True if cube-responses include a weighted-count measure."""
        return self._cubes[0].has_weighted_counts

    @lazyproperty
    def is_ca_as_0th(self):
        """True for multi-cube when first cube represents a categorical-array.

        A "CA-as-0th" tabbook tab is "3D" in the sense it is "sliced" into one table
        (partition-set) for each of the CA subvariables.
        """
        # ---can only be true for multi-cube case---
        if not self._is_multi_cube:
            return False
        # ---the rest depends on the row-var cube---
        cube = self._cubes[0]
        # ---True if row-var cube is CA---
        return cube.dimension_types[0] == DT.CA_SUBVAR

    @lazyproperty
    def missing_count(self):
        """The number of missing values from first cube in this set."""
        return self._cubes[0].missing

    @lazyproperty
    def name(self):
        """str name of first cube in this set."""
        return self._cubes[0].name

    @lazyproperty
    def partition_sets(self):
        """Sequence of cube-partition collections across all cubes of this cube-set.

        This value might look like the following for a ca-as-0th tabbook, for example:

            (
                (_Strand, _Slice, _Slice),
                (_Strand, _Slice, _Slice),
                (_Strand, _Slice, _Slice),
            )

        and might often look like this for a typical slide:

            ((_Slice,))

        Each partition set represents the partitions for a single "stacked" table. A 2D
        slide has a single partition-set of a single _Slice object, as in the second
        example above. A 3D slide would have multiple partition sets, each of a single
        _Slice. A tabook will have multiple partitions in each set, the first being
        a _Strand and the rest being _Slice objects. Multiple partition sets only arise
        for a tabbook in the CA-as-0th case.
        """
        return tuple(zip(*(cube.partitions for cube in self._cubes)))

    @lazyproperty
    def population_fraction(self):
        """The filtered/unfiltered ratio for this cube-set.

        This value is required for properly calculating population on a cube where
        a filter has been applied. Returns 1.0 for an unfiltered cube. Returns `np.nan`
        if the unfiltered count is zero, which would otherwise result in
        a divide-by-zero error.
        """
        return self._cubes[0].population_fraction

    @lazyproperty
    def n_responses(self):
        """Total number of responses considered from first cube in this set."""
        return self._cubes[0].n_responses

    @lazyproperty
    def valid_counts_summary(self):
        """The valid count summary values from first cube in this set."""
        return self._cubes[0].valid_counts_summary

    @lazyproperty
    def _cubes(self):
        """Sequence of Cube objects containing data for this analysis."""

        def iter_cubes():
            """Generate a Cube object for each of cube_responses.

            0D cube-responses and 1D second-and-later cubes are "inflated" to add their
            missing row dimension.
            """
            for idx, cube_response in enumerate(self._cube_responses):
                cube = Cube(
                    cube_response,
                    cube_idx=idx if self._is_multi_cube else None,
                    transforms=self._transforms_dicts[idx],
                    population=self._population,
                    mask_size=self._min_base,
                )
                # --- numeric-measures cubes require inflation to restore their
                # --- rows-dimension, others don't
                yield cube.inflate() if self._is_numeric_measure else cube

        return tuple(iter_cubes())

    @lazyproperty
    def _is_multi_cube(self):
        """True if more than one cube-response was provided on construction."""
        return len(self._cube_responses) > 1

    @lazyproperty
    def _is_numeric_measure(self):
        """True when CubeSet is special-case "numeric-measure" case requiring inflation.

        When a numeric variable with `mean`, `sum` or `std_dev` summary statistic
        expressed in its view, appears as the rows-dimension in a multitable analysis,
        its cube-result has been "reduced" to the mean-value of those numerics. This is
        in contrast to being "bucketized" into an arbitrary set of numeric-range
        categories like 0-5, 5-10, etc. In the process, as an artifact of the ZZ9 query
        response, that dimension is removed. As a result, the rows-dimension cube is 0D
        and the column-dimension cubes are 1D. These need to be "inflated" to restore
        the lost dimension such that they are uniform with other cube-results and can be
        processed without special-case code.

        "Inflation" is basically prefixing "1 x" to the dimensionality, for example a 1D
        of size 5 becomes a 1 x 5 2D result. Note this requires no mapping in the actual
        values because 5 = 1 x 5 = 5 (values).
        """
        # --- this case only arises in a multitable analysis ---
        if not self._is_multi_cube:
            return False

        # --- We need the cube to tell us the dimensionality. This redundant
        # --- construction is low-overhead because all Cube properties are lazy.
        return Cube(self._cube_responses[0]).ndim == 0


class Cube(object):
    """Provides access to individual slices on a cube-result.

    It also provides some attributes of the overall cube-result.

    `cube_idx` must be `None` (or omitted) for a single-cube CubeSet. This indicates the
    CubeSet contains only a single cube and influences behaviors like CA-as-0th.
    """

    def __init__(
        self, response, cube_idx=None, transforms=None, population=None, mask_size=0
    ):
        self._cube_response_arg = response
        self._transforms_dict = {} if transforms is None else transforms
        self._cube_idx_arg = cube_idx
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
            return super(Cube, self).__repr__()

    @lazyproperty
    def available_measures(self):
        """frozenset of available CUBE_MEASURE members in the cube response."""
        cube_measures = self._cube_response.get("result", {}).get("measures", {}).keys()
        return frozenset(CUBE_MEASURE(m) for m in cube_measures)

    @lazyproperty
    def counts(self):
        return self.counts_with_missings[self._valid_idxs]

    @lazyproperty
    def counts_with_missings(self):
        """ndarray of weighted, unweighted or valid counts including missing values.

        The difference from .counts is that this property includes value for missing
        categories.
        """
        return (
            self._measures.unweighted_valid_counts.raw_cube_array
            if self._measures.unweighted_valid_counts is not None
            else self._measures.weighted_counts.raw_cube_array
            if self.has_weighted_counts
            else self._measures.unweighted_counts.raw_cube_array
        )

    @lazyproperty
    def covariance(self):
        """Optional float64 ndarray of the cube_covariance if the measure exists."""
        if self._measures.covariance is None:
            return None
        return self._measures.covariance.raw_cube_array[self._valid_idxs].astype(
            np.float64
        )

    @lazyproperty
    def cube_index(self):
        """Offset of this cube within its CubeSet."""
        return 0 if self._cube_idx_arg is None else self._cube_idx_arg

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
        """_ApparentDimensions object providing access to visible dimensions.

        A cube involving a multiple-response (MR) variable has two dimensions
        for that variable (subvariables and categories dimensions), but is
        "collapsed" into a single effective dimension for cube-user purposes
        (its categories dimension is supressed). This collection will contain
        a single dimension for each MR variable and therefore may have fewer
        dimensions than appear in the cube response.
        """
        return self._all_dimensions.apparent_dimensions

    def inflate(self):
        """Return new Cube object with rows-dimension added.

        A multi-cube (tabbook) response formed from a function (e.g. mean()) on
        a numeric variable arrives without a rows-dimension.
        """
        cube_dict = self._cube_dict
        dimensions = cube_dict["result"]["dimensions"]
        default_name = "-".join([m.value for m in self._available_numeric_measures])
        # --- The default value in case of numeric variable is the combination of all
        # --- the measures expressed in the cube response.
        alias = self._numeric_measure_references.get("alias", default_name)
        name = self._numeric_measure_references.get("name", default_name).title()
        rows_dimension = {
            "references": {"alias": alias, "name": name},
            "type": {
                "categories": [{"id": 1, "name": name}],
                "class": "categorical",
            },
        }
        dimensions.insert(0, rows_dimension)
        return Cube(
            cube_dict,
            self._cube_idx_arg,
            self._transforms_dict,
            self._population,
            self._mask_size,
        )

    @lazyproperty
    def has_weighted_counts(self):
        """True if cube response has weighted count data."""
        return self.weighted_counts is not None

    @lazyproperty
    def means(self):
        """Optional float64 ndarray of the cube_means if the measure exists."""
        if self._measures.means is None:
            return None
        return self._measures.means.raw_cube_array[self._valid_idxs].astype(np.float64)

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
    def n_responses(self):
        """Total (int) number of responses considered."""
        return self._cube_response["result"].get("n", 0)

    @lazyproperty
    def overlaps(self):
        """Optional float64 ndarray of cube_overlaps if the measure exists.

        The array has as many dimensions as there are defined in the cube query, plus
        the extra subvariables dimension as the last dimension.
        """
        if self._measures.overlaps is None:
            return None
        return self._measures.overlaps.raw_cube_array[self._valid_idxs].astype(
            np.float64
        )

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
    def population_fraction(self):
        """The filtered/unfiltered ratio for cube response.

        This value is required for properly calculating population on a cube
        where a filter has been applied. Returns 1.0 for an unfiltered cube.
        Returns `np.nan` if the unfiltered count is zero, which would
        otherwise result in a divide-by-zero error.
        """
        return self._measures.population_fraction

    @lazyproperty
    def stddev(self):
        """Optional float64 ndarray of the cube_stddev if the measure exists."""
        if self._measures.stddev is None:
            return None
        return self._measures.stddev.raw_cube_array[self._valid_idxs].astype(np.float64)

    @lazyproperty
    def sums(self):
        """Optional float64 ndarray of the cube_sum if the measure exists."""
        if self._measures.sums is None:
            return None
        return self._measures.sums.raw_cube_array[self._valid_idxs].astype(np.float64)

    @lazyproperty
    def title(self):
        """str alternate-name given to cube-result.

        This value is suitable for naming a Strand when displayed as a column. In this
        use-case it is a stand-in for the columns-dimension name since a strand has no
        columns dimension.
        """
        return self._cube_dict["result"].get("title", "Untitled")

    @lazyproperty
    def unweighted_counts(self):
        """ndarray of unweighted counts, valid elements only.

        Unweighted counts are drawn from the `result.counts` field of the cube result.
        These counts are always present, even when the measure is numeric and there are
        no count measures. These counts are always unweighted, regardless of whether the
        cube is "weighted".

        In case of presence of valid counts in the cube response the counts are replaced
        with the valid counts measure.
        """
        unweighted_counts = (
            self._measures.unweighted_valid_counts
            if self._measures.unweighted_valid_counts is not None
            else self._measures.unweighted_counts
        )
        return unweighted_counts.raw_cube_array[self._valid_idxs]

    @lazyproperty
    def unweighted_valid_counts(self):
        """Optional float64 ndarray of unweighted_valid_counts if the measure exists."""
        if self._measures.unweighted_valid_counts is None:
            return None
        return self._measures.unweighted_valid_counts.raw_cube_array[
            self._valid_idxs
        ].astype(np.float64)

    @lazyproperty
    def valid_counts_summary(self):
        """Optional ndarray of summary valid counts"""
        if not self._measures.unweighted_valid_counts:
            return None
        # --- In case of ndim >= 2 the sum should be done on the second axes to get
        # --- the correct sequence of valid count (e.g. CA_SUBVAR).
        axis = 1 if len(self._all_dimensions) >= 2 else 0
        return np.sum(
            self._measures.unweighted_valid_counts.raw_cube_array[self._valid_idxs],
            axis=axis,
        )

    @lazyproperty
    def valid_overlaps(self):
        """Optional float64 ndarray of cube_valid_overlaps if the measure exists.

        The array has as many dimensions as there are defined in the cube query, plus
        the extra subvariables dimension as the last dimension.
        """
        if self._measures.valid_overlaps is None:
            return None  # pragma: no cover
        return self._measures.valid_overlaps.raw_cube_array[self._valid_idxs].astype(
            np.float64
        )

    @lazyproperty
    def weighted_counts(self):
        """ndarray of weighted counts, valid elements only.

        In case of presence of valid counts in the cube response the weighted counts
        are replaced with the valid counts measure.
        """
        weighted_counts = (
            self._measures.weighted_valid_counts
            if self._measures.weighted_valid_counts is not None
            else self._measures.weighted_counts
        )
        return (
            weighted_counts.raw_cube_array[self._valid_idxs]
            if weighted_counts is not None
            else None
        )

    @lazyproperty
    def weighted_valid_counts(self):
        """Optional float64 ndarray of weighted_valid_counts if the measure exists."""
        if self._measures.weighted_valid_counts is None:
            return None
        return self._measures.weighted_valid_counts.raw_cube_array[
            self._valid_idxs
        ].astype(np.float64)

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
    def _available_numeric_measures(self):
        """tuple of available numeric measures expressed in the cube_response.

        Basically the numeric measures are the intersection between all the measures
        within the cube response and the defined NUMERIC_MEASURES.
        """
        return tuple(self.available_measures.intersection(NUMERIC_MEASURES))

    @lazyproperty
    def _ca_as_0th(self):
        """True if slicing is to be performed in so-called "CA-as-0th" mode.

        In this mode, a categorical-array (CA) cube (2D) is sliced into a sequence of 1D
        slices, each of which represents one subvariable of the CA variable. Normally,
        a 2D cube-result becomes a single slice.
        """
        return (
            (self._cube_idx_arg == 0 or self._is_single_filter_col_cube)
            and len(self.dimension_types) > 0
            and self.dimension_types[0] == DT.CA
        )

    @lazyproperty
    def _cube_dict(self):
        """dict containing raw cube response, parsed from JSON payload."""
        cube_dict = copy.deepcopy(self._cube_response)
        if self._numeric_measure_subvariables:
            dimensions = cube_dict.get("result", {}).get("dimensions", [])
            # ---dim inflation---
            # ---In case of numeric arrays, we need to inflate the row dimension
            # ---according to the mean subvariables. For each subvar the row dimension
            # ---will have a new element related to the subvar metadata.
            dimensions.insert(0, self._numeric_array_dimension)
        return cube_dict

    @lazyproperty
    def _cube_response(self):
        """dict representing the parsed cube response arguments."""
        try:
            response = self._cube_response_arg
            # ---parse JSON to a dict when constructed with JSON---
            cube_response = (
                response if isinstance(response, dict) else json.loads(response)
            )
            # ---cube is 'value' item in a shoji response---
            return cube_response.get("value", cube_response)
        except TypeError:
            raise TypeError(
                "Unsupported type <%s> provided. Cube response must be JSON "
                "(str) or dict." % type(self._cube_response_arg).__name__
            )

    @lazyproperty
    def _is_single_filter_col_cube(self):
        """bool determines if it is a single column filter cube."""
        return self._cube_dict["result"].get("is_single_col_cube", False)

    @lazyproperty
    def _measures(self):
        """_Measures object for this cube.

        Provides access to count based measures and numeric measures (e.g. mean, sum)
        when available.
        """
        return _Measures(self._cube_dict, self._all_dimensions, self._cube_idx_arg)

    @lazyproperty
    def _numeric_measure_references(self):
        """Dict of numeric measure references, typically for numeric measures."""
        if not self._available_numeric_measures:
            return {}
        cube_response = self._cube_response
        cube_measures = cube_response.get("result", {}).get("measures", {})
        metadata = cube_measures.get(self._available_numeric_measures[0].value, {}).get(
            "metadata", {}
        )
        return metadata.get("references", {})

    @lazyproperty
    def _numeric_measure_subvariables(self):
        """List of mean subvariables, typically for numeric arrays."""
        if not self._available_numeric_measures:
            return []
        cube_response = self._cube_response
        cube_measures = cube_response.get("result", {}).get("measures", {})
        metadata = cube_measures.get(self._available_numeric_measures[0].value, {}).get(
            "metadata", {}
        )
        return metadata.get("type", {}).get("subvariables", [])

    @lazyproperty
    def _numeric_array_dimension(self):
        """Rows dimension object according to the numeric-measure subvariables."""
        if not self._numeric_measure_subvariables:
            return None
        subrefs = self._numeric_measure_references.get("subreferences", [])
        rows_dimension = {
            "references": {
                "alias": self._numeric_measure_references.get("alias"),
                "name": self._numeric_measure_references.get("name"),
            },
            "type": {"elements": [], "class": "enum", "subtype": {"class": "num_arr"}},
        }
        # ---In case of numeric arrays the row dimension should contains additional
        # ---information related to the subreferences for each subvariable of the
        # ---array.
        for i, _ in enumerate(self._numeric_measure_subvariables):
            # ---The row dimensions elements must be expanded with the alias and the
            # ---name of the numeric array mean measure subreferences.
            rows_dimension["type"].get("elements", []).append(
                {
                    "id": i,
                    "value": {
                        "references": {
                            "alias": subrefs[i].get("alias") if subrefs else None,
                            "name": subrefs[i].get("name") if subrefs else None,
                        },
                        "id": self._numeric_measure_subvariables[i],
                    },
                },
            )
        return rows_dimension

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
        """Tuple of int64 ndarrays of the valid elements idx for each dimension."""
        valid_idxs = np.ix_(
            *tuple(d.valid_elements.element_idxs for d in self._all_dimensions)
        )
        # The dimension dimension order can change in case of numeric array variable on
        # the row, and so valid indices needs to be returned in an ordered way.
        return tuple(valid_idxs[i] for i in self._all_dimensions.dimension_order)


class _Measures(object):
    """Provides access to measures contained in cube response."""

    def __init__(self, cube_dict, all_dimensions, cube_idx_arg=None):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions
        self._cube_idx_arg = cube_idx_arg

    @lazyproperty
    def covariance(self):
        """Optional _CovarianceMeasure object providing access to covariance values.

        Will be None if covariance is not available int the cube response.
        """
        covariance = _CovarianceMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return None if covariance.raw_cube_array is None else covariance

    @lazyproperty
    def means(self):
        """Optional _MeanMeasure object providing access to means values.

        Will be None if no means are available on the counts.
        """
        mean = _MeanMeasure(self._cube_dict, self._all_dimensions, self._cube_idx_arg)
        return None if mean.raw_cube_array is None else mean

    @lazyproperty
    def missing_count(self):
        """numeric representing count of missing rows in cube response."""
        if self.unweighted_valid_counts is not None:
            return self.unweighted_valid_counts.missing_count
        # The check on the means measure is needed for retro-compatibility with the old
        # fixtures that don't have valid_counts.
        if self.means is not None:
            return self.means.missing_count
        return self._cube_dict["result"].get("missing", 0)

    @lazyproperty
    def overlaps(self):
        """Optional _OverlapMeasure object providing access to overlaps values.

        Will be None if no overlaps are available on the cube result.
        """
        overlap = _OverlapMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return None if overlap.raw_cube_array is None else overlap

    @lazyproperty
    def population_fraction(self):
        """The filtered/unfiltered ratio for cube response.

        The filtered counts are calculated for complete-cases. This means that only the
        non-missing entries are included in the filtered counts. Complete cases are
        used only if the corresponding cases are included in the cube response. If not,
        the old-style default calculation is used.

        This value is required for properly calculating population on a cube
        where a filter has been applied. Returns 1.0 for an unfiltered cube.
        Returns `np.nan` if the unfiltered count is zero, which would
        otherwise result in a divide-by-zero error.
        """

        # Try and get the new-style complete-cases filtered counts
        filter_stats = (
            self._cube_dict["result"]
            .get("filter_stats", {})
            .get("filtered_complete", {})
            .get("weighted")
        )
        if filter_stats:
            # If new format is present in response json, use that for pop fraction
            numerator = filter_stats["selected"]
            denominator = numerator + filter_stats["other"]
        else:
            # If new format is not available, default to old-style calculation
            numerator = self._cube_dict["result"].get("filtered", {}).get("weighted_n")
            denominator = (
                self._cube_dict["result"].get("unfiltered", {}).get("weighted_n")
            )

        try:
            return numerator / denominator
        except ZeroDivisionError:
            return np.nan
        except Exception:
            return 1.0

    @lazyproperty
    def stddev(self):
        """_StdDevMeasure object providing access to cube stddev values.

        None when the cube response does not contain a stddev measure.
        """
        stddev = _StdDevMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return None if stddev.raw_cube_array is None else stddev

    @lazyproperty
    def sums(self):
        """_SumMeasure object providing access to cube sum values.

        None when the cube response does not contain a sum measure.
        """
        sums = _SumMeasure(self._cube_dict, self._all_dimensions, self._cube_idx_arg)
        return None if sums.raw_cube_array is None else sums

    @lazyproperty
    def unweighted_counts(self):
        """_UnweightedCountMeasure object for this cube.

        This object provides access to unweighted counts for this cube,
        whether or not the cube contains weighted counts.
        """
        return _UnweightedCountMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )

    @lazyproperty
    def unweighted_valid_counts(self):
        """_UnweightedValidCountsMeasure object for this cube.

        Can be None when cube doesn't have unweighted valid counts.
        """
        valid_counts = _UnweightedValidCountsMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return valid_counts if valid_counts.raw_cube_array is not None else None

    @lazyproperty
    def valid_overlaps(self):
        """Optional _ValidOverlapMeasure object providing access to valid overlaps vals.

        Will be None if no valid overlaps are available on the cube result.
        """
        overlap = _ValidOverlapMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return None if overlap.raw_cube_array is None else overlap

    @lazyproperty
    def weighted_counts(self):
        """Optional _WeightedCountMeasure object for this cube.

        Can be None when the cube is unweighted.
        """
        weighted_counts = _WeightedCountMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return weighted_counts if weighted_counts.raw_cube_array is not None else None

    @lazyproperty
    def weighted_valid_counts(self):
        """_WeightedValidCountsMeasure object for this cube.

        Can be None when cube doesn't have weighted valid counts.
        """
        valid_counts = _WeightedValidCountsMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )
        return valid_counts if valid_counts.raw_cube_array is not None else None


class _BaseMeasure(object):
    """Base class for measure objects."""

    def __init__(self, cube_dict, all_dimensions, cube_idx_arg=None):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions
        self._cube_idx_arg = cube_idx_arg

    @lazyproperty
    def raw_cube_array(self):
        """Optional read-only ndarray of measure values from cube-response.

        The shape of the ndarray mirrors the shape of the (raw) cube
        response. Specifically, it includes values for missing elements, any
        MR_CAT dimensions, and any prunable rows and columns. Returns None
        if the measure is not available in cube.
        """
        if self._flat_values is None:
            return None
        raw_cube_array = self._flat_values.reshape(self._shape)
        # ---must be read-only to avoid hard-to-find bugs---
        raw_cube_array.flags.writeable = False
        return raw_cube_array

    @lazyproperty
    def _flat_values(self):  # pragma: no cover
        """Return ndarray of np.float64 values as found in cube response.

        This property must be implemented by each subclass.
        """
        raise NotImplementedError("must be implemented by each subclass")

    @lazyproperty
    def _shape(self):
        """tuple(int) representing the shape of the raw-cube measure array.

        If needed, this property can be overridden, to accustom different measure shapes
        even if the basic cube has the same original shape.
        """
        return self._all_dimensions.shape


class _CovarianceMeasure(_BaseMeasure):
    """Covariance values from a cube-response."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 cov values as found in cube response.

        Covariance data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        if self._measure_payload is None:
            return None
        return np.array(
            tuple(
                np.nan if type(x) is dict else x for x in self._measure_payload["data"]
            ),
            dtype=np.float64,
        ).flatten()

    @lazyproperty
    def _measure_payload(self):
        """dict representing the covariance measure part of the cube response."""
        return self._cube_dict["result"].get("measures", {}).get("covariance")

    @lazyproperty
    def _numeric_measure_subvariables(self):
        """List of subvariables, typically for numeric arrays."""
        metadata = self._measure_payload.get("metadata", {})
        return metadata.get("type", {}).get("subvariables", [])

    @lazyproperty
    def _shape(self):
        """tuple(int) representing the shape of the covariance."""
        return self._all_dimensions.shape + (len(self._numeric_measure_subvariables),)


class _MeanMeasure(_BaseMeasure):
    """Statistical mean values from a cube-response."""

    @lazyproperty
    def missing_count(self):
        """Numeric value representing count of missing rows in response."""
        return self._cube_dict["result"]["measures"]["mean"].get("n_missing", 0)

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 mean values as found in cube response.

        Mean data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        measure_payload = self._cube_dict["result"].get("measures", {}).get("mean")
        if measure_payload is None:
            return None
        return np.array(
            tuple(np.nan if type(x) is dict else x for x in measure_payload["data"]),
            dtype=np.float64,
        ).flatten()


class _OverlapMeasure(_BaseMeasure):
    """Overlap values from a cube-response."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 overlap values as found in cube response.

        Overlap data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        if self._measure_payload is None:
            return None
        return np.array(
            tuple(
                np.nan if type(x) is dict else x for x in self._measure_payload["data"]
            ),
            dtype=np.float64,
        ).flatten()

    @lazyproperty
    def _measure_payload(self):
        """dict representing the overlaps measure part of the cube response."""
        return self._cube_dict["result"].get("measures", {}).get("overlap")

    @lazyproperty
    def _shape(self):
        """tuple(int) representing shape of the overlaps measure.

        The overlaps measure is characteristic in that it produces an additional
        dimension for the Multiple Response subvariables. That dimension is always
        found at the end of the result shape, because of how responses are
        generated by the backend (ZZ9) mechanism.
        """
        n_subvars = len(self._measure_payload["metadata"]["type"]["subvariables"])
        return self._all_dimensions.shape + (n_subvars,)


class _StdDevMeasure(_BaseMeasure):
    """Statistical stddev values from a cube-response."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D float64 ndarray of stddev values as found in cube response.

        StdDev data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        measure_payload = self._cube_dict["result"].get("measures", {}).get("stddev")
        if measure_payload is None:
            return None

        return np.array(
            tuple(np.nan if type(x) is dict else x for x in measure_payload["data"]),
            dtype=np.float64,
        ).flatten()


class _SumMeasure(_BaseMeasure):
    """Statistical sum values from a cube-response."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D float64 ndarray of sum values as found in cube response.

        Sum data may include missing items represented by a dict like
        {'?': -1} in the cube response. These are replaced by np.nan in the
        returned value.
        """
        measure_payload = self._cube_dict["result"].get("measures", {}).get("sum")
        if measure_payload is None:
            return None

        return np.array(
            tuple(np.nan if type(x) is dict else x for x in measure_payload["data"]),
            dtype=np.float64,
        ).flatten()


class _UnweightedCountMeasure(_BaseMeasure):
    """Unweighted counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """1D np.ndarray of np.float64 counts before weighting.

        Use np.float64s to avoid int overflow bugs and so we can use nan.
        """
        return np.array(self._cube_dict["result"]["counts"], dtype=np.float64)


class _UnweightedValidCountsMeasure(_BaseMeasure):
    """Unweighted Valid counts for cube."""

    @lazyproperty
    def missing_count(self):
        """numeric representing count of missing rows reflected in response."""
        return self._cube_dict["result"]["measures"]["valid_count_unweighted"].get(
            "n_missing", 0
        )

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 unweighted valid counts."""
        valid_counts = (
            self._cube_dict["result"]["measures"]
            .get("valid_count_unweighted", {})
            .get("data", [])
        )
        return np.array(valid_counts, dtype=np.float64) if valid_counts else None


class _ValidOverlapMeasure(_OverlapMeasure):
    """Valid overlap values from a cube-response."""

    @lazyproperty
    def _measure_payload(self):
        """dict representing the valid overlaps measure part of the cube response."""
        return self._cube_dict["result"].get("measures", {}).get("valid_overlap")


class _WeightedCountMeasure(_BaseMeasure):
    """Weighted counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 numeric counts after weighting."""
        unweighted_counts = self._cube_dict["result"]["counts"]
        weighted_counts = (
            self._cube_dict["result"]["measures"].get("count", {}).get("data")
        )
        if unweighted_counts == weighted_counts or weighted_counts is None:
            return None

        return np.array(weighted_counts, dtype=np.float64)


class _WeightedValidCountsMeasure(_BaseMeasure):
    """Weighted Valid counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """Optional 1D np.ndarray of np.float64 weighted valid counts."""
        valid_counts = (
            self._cube_dict["result"]["measures"]
            .get("valid_count_weighted", {})
            .get("data", [])
        )
        return np.array(valid_counts, dtype=np.float64) if valid_counts else None
