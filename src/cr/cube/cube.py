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
from cr.cube.enums import DIMENSION_TYPE as DT
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
    def has_means(self):
        """True if cubes in this set include a means measure."""
        return self._cubes[0].has_means

    @lazyproperty
    def has_weighted_counts(self):
        """True if cube-responses include a weighted-count measure."""
        return self._cubes[0].is_weighted

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
                # --- numeric-mean cubes require inflation to restore their
                # --- rows-dimension, others don't
                yield cube.inflate() if self._is_numeric_mean else cube

        return tuple(iter_cubes())

    @lazyproperty
    def _is_multi_cube(self):
        """True if more than one cube-response was provided on construction."""
        return len(self._cube_responses) > 1

    @lazyproperty
    def _is_numeric_mean(self):
        """True when CubeSet is special-case "numeric-mean" case requiring inflation.

        When a numeric variable appears as the rows-dimension in a multitable analysis,
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
    def counts(self):
        return self.counts_with_missings[self._valid_idxs]

    @lazyproperty
    def counts_with_missings(self):
        return self._measure(self.is_weighted).raw_cube_array

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

    @lazyproperty
    def has_means(self):
        """True if cube includes a means measure."""
        return self._measures.means is not None

    def inflate(self):
        """Return new Cube object with rows-dimension added.

        A multi-cube (tabbook) response formed from a function (e.g. mean()) on
        a numeric variable arrives without a rows-dimension.
        """
        cube_dict = self._cube_dict
        dimensions = cube_dict["result"]["dimensions"]
        rows_dimension = {
            "references": {"alias": "mean", "name": "mean"},
            "type": {
                "categories": [{"id": 1, "name": "Mean"}],
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
    def title(self):
        """str alternate-name given to cube-result.

        This value is suitable for naming a Strand when displayed as a column. In this
        use-case it is a stand-in for the columns-dimension name since a strand has no
        columns dimension.
        """
        return self._cube_dict["result"].get("title", "Untitled")

    @lazyproperty
    def n_responses(self):
        """Total (int) number of responses considered."""
        return self._cube_response["result"].get("n", 0)

    @lazyproperty
    def unweighted_counts(self):
        """ndarray of unweighted counts, valid elements only.

        Unweighted counts are drawn from the `result.counts` field of the cube result.
        These counts are always present, even when the measure is `mean` and there are
        no count measures. These counts are always unweighted, regardless of whether the
        cube is "weighted".
        """
        return self._measures.unweighted_counts.raw_cube_array[self._valid_idxs]

    @lazyproperty
    def valid_counts(self):
        """ndarray of valid counts, valid elements only."""
        valid_counts = self._measure(self.is_weighted).valid_counts
        if valid_counts.any():
            return valid_counts[self._valid_idxs]
        return np.empty(0)

    @lazyproperty
    def valid_counts_summary(self):
        """ndarray of summary valid counts"""
        if self.valid_counts.any():
            # --- In case of ndim > 2 the sum should be done on the second axes to get
            # --- the correct sequence of valid count (e.g. CA_SUBVAR).
            axis = 1 if len(self._all_dimensions) > 2 else 0
            return np.sum(self.valid_counts, axis=axis)
        return np.empty(0)

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
        return (
            (self._cube_idx_arg == 0 or self._is_single_filter_col_cube)
            and len(self.dimension_types) > 0
            and self.dimension_types[0] == DT.CA
        )

    @lazyproperty
    def _cube_dict(self):
        """dict containing raw cube response, parsed from JSON payload."""
        cube_dict = copy.deepcopy(self._cube_response)
        if self._mean_subvariables:
            dimensions = cube_dict.get("result", {}).get("dimensions", [])
            # ---cube inflation---
            # ---In case of numeric arrays, we need to inflate the columns dimension
            # ---according to the mean subvariables. For each subvar the col dimension
            # ---will have a new element related to the subvar metadata.
            if self._cube_idx_arg:
                dimensions.insert(0, self._numeric_array_dimension)
            else:
                dimensions.append(self._numeric_array_dimension)
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
    def _mean_references(self):
        """Dict of mean references, tipically for numeric arrays."""
        cube_response = self._cube_response
        cube_measures = cube_response.get("result", {}).get("measures", {})
        metadata = cube_measures.get("mean", {}).get("metadata", {})
        return metadata.get("references", {})

    @lazyproperty
    def _mean_subvariables(self):
        """List of mean subvariables, tipically for numeric arrays."""
        cube_response = self._cube_response
        cube_measures = cube_response.get("result", {}).get("measures", {})
        metadata = cube_measures.get("mean", {}).get("metadata", {})
        return metadata.get("type", {}).get("subvariables", [])

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
        return _Measures(self._cube_dict, self._all_dimensions, self._cube_idx_arg)

    @lazyproperty
    def _numeric_array_dimension(self):
        """Column dimension object according to the mean subvariables."""
        if not self._mean_subvariables:
            return None
        subrefs = self._mean_references.get("subreferences", [])
        column_dimension = {
            "references": {
                "alias": self._mean_references.get("alias", "mean"),
                "name": self._mean_references.get("name", "mean"),
            },
            "type": {"elements": [], "class": "enum", "subtype": {"class": "num_arr"}},
        }
        # ---In case of numeric arrays the column dimension should contains additional
        # ---information related to the subreferences for each subvariable of the
        # ---array.
        for i, _ in enumerate(self._mean_subvariables):
            # ---The column dimensions elements must be expanded with the alias and the
            # ---name of the numeric array mean measure subreferences.
            column_dimension["type"].get("elements", []).append(
                {
                    "id": i,
                    "value": {
                        "references": {
                            "alias": subrefs[i].get("alias") if subrefs else None,
                            "name": subrefs[i].get("name") if subrefs else None,
                        },
                        "id": self._mean_subvariables[i],
                    },
                },
            )
        return column_dimension

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
        valid_idxs = np.ix_(
            *tuple(d.valid_elements.element_idxs for d in self._all_dimensions)
        )

        def _reshape_idxs(valid_indices):
            if self._measure(self.is_weighted).requires_array_transposition:
                if len(self._all_dimensions) == 3:
                    # ---In case of 3D array and a numeric array is involved we have to
                    # ---change the order of the valid idxs, from [0,1,2] to [1,2,0].
                    # ---This way to reshape the valid index will be changed when the
                    # ---dim_order option will be available within the cube dict.
                    return valid_indices[1], valid_indices[-1], valid_indices[0]
                # ---NOTE FOR FUTURE: We'll need a way to tell to the cube which
                # ---dimensions we want transposed and which not. So the ::-1 can be
                # ---converted in something more robust.
                return valid_indices[::-1]
            return valid_indices

        return _reshape_idxs(valid_idxs)


class _Measures(object):
    """Provides access to measures contained in cube response."""

    def __init__(self, cube_dict, all_dimensions, cube_idx_arg=None):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions
        self._cube_idx_arg = cube_idx_arg

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
        return (
            _MeanMeasure(self._cube_dict, self._all_dimensions, self._cube_idx_arg)
            if mean_measure_dict
            else None
        )

    @lazyproperty
    def missing_count(self):
        """numeric representing count of missing rows in cube response."""
        if self.means:
            return self.means.missing_count
        return self._cube_dict["result"].get("missing", 0)

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
    def unweighted_counts(self):
        """_UnweightedCountMeasure object for this cube.

        This object provides access to unweighted counts for this cube,
        whether or not the cube contains weighted counts.
        """
        return _UnweightedCountMeasure(
            self._cube_dict, self._all_dimensions, self._cube_idx_arg
        )

    @lazyproperty
    def weighted_counts(self):
        """_WeightedCountMeasure object for this cube.

        This object provides access to weighted counts for this cube, if
        available. If the cube response is not weighted, the
        _UnweightedCountMeasure object for this cube is returned.
        """
        return (
            _WeightedCountMeasure(
                self._cube_dict, self._all_dimensions, self._cube_idx_arg
            )
            if self.is_weighted
            else _UnweightedCountMeasure(
                self._cube_dict, self._all_dimensions, self._cube_idx_arg
            )
        )


class _BaseMeasure(object):
    """Base class for measure objects."""

    def __init__(self, cube_dict, all_dimensions, cube_idx_arg=None):
        self._cube_dict = cube_dict
        self._all_dimensions = all_dimensions
        self._cube_idx_arg = cube_idx_arg

    @lazyproperty
    def raw_cube_array(self):
        """Return read-only ndarray of measure values from cube-response.

        The shape of the ndarray mirrors the shape of the (raw) cube
        response. Specifically, it includes values for missing elements, any
        MR_CAT dimensions, and any prunable rows and columns.
        """
        raw_cube_array = np.array(self._flat_values).flatten().reshape(self._shape)
        # ---must be read-only to avoid hard-to-find bugs---
        raw_cube_array.flags.writeable = False
        return raw_cube_array

    @lazyproperty
    def requires_array_transposition(self):
        """True if raw cube array needs transposition, False otherwise.

        When a Cube is part of a cubeset and one of the dimension type is a NUM_ARRAY,
        it's cube_array values have to be transposed.
        This business rule needs to drive correctly tabbook exports when a numeric array
        dimension is expressed.
        """
        return (
            self._cube_idx_arg is not None
            and len(self._all_dimensions) >= 2
            and any([d.dimension_type == DT.NUM_ARRAY for d in self._all_dimensions])
        )

    @lazyproperty
    def valid_counts(self):
        """np.array of valid count measure in the cube response."""
        valid_counts = (
            self._cube_dict["result"]["measures"]
            .get("valid_count_unweighted", {})
            .get("data", [])
        )
        if not valid_counts:
            return np.empty(0)
        valid_counts_array = np.array(valid_counts).reshape(self._all_dimensions.shape)
        return (
            valid_counts_array.T
            if self.requires_array_transposition
            else valid_counts_array
        )

    @lazyproperty
    def _flat_values(self):  # pragma: no cover
        """Return tuple of mean values as found in cube response.

        This property must be implemented by each subclass.
        """
        raise NotImplementedError("must be implemented by each subclass")

    @lazyproperty
    def _shape(self):
        """All dimensions shape (row, col)"""
        # NOTE: Inverting the shape cannot be enough in future when we'll have more than
        # 2 dimensions in the new dim_order option.
        return (
            self._all_dimensions.shape[::-1]
            if self.requires_array_transposition
            else self._all_dimensions.shape
        )


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
        if (
            self._cube_dict["result"]["measures"]
            .get("valid_count_unweighted", {})
            .get("data")
        ):
            # ---If valid_count are expressed in the cube dict, returns its data.
            # ---This condition can happen in case of numeric array cube response.
            # ---Under this circumstances the numeric array measures will contain the
            # ---mean measure and a valid count measure for the unweighted counts.
            return tuple(
                self._cube_dict["result"]["measures"]["valid_count_unweighted"]["data"]
            )
        return tuple(self._cube_dict["result"]["counts"])


class _WeightedCountMeasure(_BaseMeasure):
    """Weighted counts for cube."""

    @lazyproperty
    def _flat_values(self):
        """tuple of numeric counts after weighting."""
        return tuple(self._cube_dict["result"]["measures"]["count"]["data"])
