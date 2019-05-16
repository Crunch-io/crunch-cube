# encoding: utf-8

from __future__ import division

from collections import namedtuple
import numpy as np
from scipy.stats.contingency import expected_freq
from scipy.stats import norm

from cr.cube.dimension import NewDimension
from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.frozen_min_base_size_mask import MinBaseSizeMask
from cr.cube.util import lazyproperty
from cr.cube.measures.new_pairwise_significance import NewPairwiseSignificance


class FrozenSlice(object):
    """Main point of interaction with the outer world."""

    def __init__(
        self,
        cube,
        slice_idx=0,
        weighted=True,
        transforms=None,
        population=None,
        ca_as_0th=None,
        mask_size=0,
    ):
        self._cube = cube
        self._slice_idx = slice_idx
        self._transforms_dict = {} if transforms is None else transforms
        self._weighted = weighted
        self._population = population
        self._ca_as_0th = ca_as_0th
        self._mask_size = mask_size

    # ---interface ---------------------------------------------------

    @lazyproperty
    def insertion_rows_idxs(self):
        return self._calculator.insertion_rows_idxs

    @lazyproperty
    def insertion_columns_idxs(self):
        return self._calculator.insertion_columns_idxs

    @lazyproperty
    def name(self):
        return self.rows_dimension_name

    @lazyproperty
    def table_name(self):
        if self._cube.ndim < 3 and not self._ca_as_0th:
            return None

        title = self._cube.name
        table_name = self._cube.labels()[0][self._slice_idx]
        return "%s: %s" % (title, table_name)

    @lazyproperty
    def dimension_types(self):
        return tuple(dimension.dimension_type for dimension in self.dimensions)

    @lazyproperty
    def ndim(self):
        _1D_slice_types = (_1DMrWithMeansSlice, _1DMeansSlice, _1DCatSlice, _1DMrSlice)
        if isinstance(self._slice, _CatXCatMeansSlice):
            return 2
        if isinstance(self._slice, _1D_slice_types):
            return 1
        elif isinstance(self._slice, _0DMeansSlice):
            return 0
        return 2

    @lazyproperty
    def pairwise_indices(self):
        alpha = self._transforms_dict.get("pairwise_indices", {}).get("alpha", 0.05)
        only_larger = self._transforms_dict.get("pairwise_indices", {}).get(
            "only_larger", True
        )
        return NewPairwiseSignificance(
            self, alpha=alpha, only_larger=only_larger
        ).pairwise_indices

    @lazyproperty
    def pairwise_significance_tests(self):
        """tuple of _ColumnPairwiseSignificance tests.

        Result has as many elements as there are columns in the slice. Each
        significance test contains `p_vals` and `t_stats` (ndarrays that represent
        probability values and statistical scores).
        """
        return tuple(
            NewPairwiseSignificance(self).values[column_idx]
            for column_idx in range(len(self._assembler.columns))
        )

    @lazyproperty
    def summary_pairwise_indices(self):
        alpha = self._transforms_dict.get("pairwise_indices", {}).get("alpha", 0.05)
        only_larger = self._transforms_dict.get("pairwise_indices", {}).get(
            "only_larger", True
        )
        return NewPairwiseSignificance(
            self, alpha=alpha, only_larger=only_larger
        ).summary_pairwise_indices

    @lazyproperty
    def scale_means_row(self):
        return self._calculator.scale_means_row

    @lazyproperty
    def scale_means_row_margin(self):
        return self._calculator.scale_means_row_margin

    @lazyproperty
    def scale_means_column_margin(self):
        return self._calculator.scale_means_column_margin

    @lazyproperty
    def scale_means_column(self):
        return self._calculator.scale_means_column

    @lazyproperty
    def column_index(self):
        """ndarray of column index percentages.

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the corresponding variable.
        """
        return self._calculator.column_index

    @lazyproperty
    def min_base_size_mask(self):
        return MinBaseSizeMask(self, self._mask_size)

    @lazyproperty
    def base_counts(self):
        return self._calculator.base_counts

    @lazyproperty
    def column_base(self):
        return self._calculator.column_base

    @lazyproperty
    def column_labels(self):
        return self._calculator.column_labels

    @lazyproperty
    def column_margin(self):
        return self._calculator.column_margin

    @lazyproperty
    def column_percentages(self):
        return self.column_proportions * 100

    @lazyproperty
    def column_proportions(self):
        return self._calculator.column_proportions

    @lazyproperty
    def columns_dimension_name(self):
        """str name assigned to rows-dimension.

        The empty string ("") for a 0D and 1D slices (until we get to all slices
        being 2D). Reflects the resolved dimension-name transform cascade.
        """
        if len(self.dimensions) < 2:
            return ""
        return self.dimensions[1].name

    @lazyproperty
    def counts(self):
        return self._calculator.counts

    @lazyproperty
    def means(self):
        return self._calculator.means

    @lazyproperty
    def names(self):
        return self._slice.names

    @lazyproperty
    def population_counts(self):
        return (
            self.table_proportions * self._population * self._cube.population_fraction
        )

    @lazyproperty
    def pvals(self):
        return self._calculator.pvals

    @lazyproperty
    def row_base(self):
        return self._calculator.row_base

    @lazyproperty
    def row_labels(self):
        return self._calculator.row_labels

    @lazyproperty
    def column_labels_with_ids(self):
        return self._calculator.column_labels_with_ids

    @lazyproperty
    def row_margin(self):
        return self._calculator.row_margin

    @lazyproperty
    def row_percentages(self):
        return self.row_proportions * 100

    @lazyproperty
    def row_proportions(self):
        return self._calculator.row_proportions

    @lazyproperty
    def rows_dimension_description(self):
        """str description assigned to rows-dimension.

        The empty string ("") for a 0D slice (until we get to all slices being 2D).
        Reflects the resolved dimension-description transform cascade.
        """
        if len(self.dimensions) == 0:
            return ""
        return self.dimensions[0].description

    @lazyproperty
    def rows_dimension_name(self):
        """str name assigned to rows-dimension.

        The empty string ("") for a 0D slice (until we get to all slices being 2D).
        Reflects the resolved dimension-name transform cascade.
        """
        if len(self.dimensions) == 0:
            return ""
        return self.dimensions[0].name

    @lazyproperty
    def rows_dimension_type(self):
        """Member of DIMENSION_TYPE enum describing type of rows dimension, or None.

        This value is None for a 0D slice (until we get to all slices being 2D).
        """
        if len(self.dimensions) == 0:
            return None
        return self.dimensions[0].dimension_type

    @lazyproperty
    def columns_dimension_type(self):
        if len(self.dimensions) < 2:
            return None
        return self.dimensions[1].dimension_type

    @lazyproperty
    def shape(self):
        return self.counts.shape

    @lazyproperty
    def table_base(self):
        return self._calculator.table_base

    @lazyproperty
    def table_margin(self):
        return self._calculator.table_margin

    @lazyproperty
    def table_base_unpruned(self):
        return self._calculator.table_base_unpruned

    @lazyproperty
    def table_margin_unpruned(self):
        return self._calculator.table_margin_unpruned

    @lazyproperty
    def table_percentages(self):
        return self.table_proportions * 100

    @lazyproperty
    def table_proportions(self):
        return self._calculator.table_proportions

    @lazyproperty
    def zscore(self):
        return self._calculator.zscore

    # ---implementation (helpers)-------------------------------------

    @lazyproperty
    def _assembler(self):
        return Assembler(self._slice, self._transforms)

    @lazyproperty
    def _calculator(self):
        return Calculator(self._assembler)

    @lazyproperty
    def dimensions(self):
        """tuple of (row,) or (row, col) Dimension objects, depending on 1D or 2D."""
        # TODO: pretty messy while we're shimming in NewDimensions, should clean up
        # pretty naturally after FrozenSlice has its own loader.

        dimensions = self._cube.dimensions[-2:]

        # ---special-case for 0D mean cube---
        if not dimensions:
            return dimensions

        if self._ca_as_0th:
            # Represent CA slice as 1-D rather than 2-D
            dimensions = (dimensions[-1],)

        rows_dimension = NewDimension(
            dimensions[0], self._transforms_dict.get("rows_dimension", {})
        )

        if len(dimensions) == 1:
            return (rows_dimension,)

        columns_dimension = NewDimension(
            dimensions[1], self._transforms_dict.get("columns_dimension", {})
        )

        return (rows_dimension, columns_dimension)

    @lazyproperty
    def _pruning(self):
        """True if any of dimensions has pruning."""
        # TODO: Implement separarte pruning for rows and columns
        return any(dimension.prune for dimension in self.dimensions)

    def _create_means_slice(self, counts, base_counts):
        if self._cube.ndim == 0:
            return _0DMeansSlice(counts, base_counts)
        elif self._cube.ndim == 1:
            if self.dimensions[0].dimension_type == DT.MR:
                return _1DMrWithMeansSlice(self.dimensions[0], counts, base_counts)
            return _1DMeansSlice(self.dimensions[0], counts, base_counts)
        elif self._cube.ndim >= 2:
            if self._cube.ndim == 3:
                base_counts = base_counts[self._slice_idx]
                counts = counts[self._slice_idx]
            if self.dimensions[0].dimension_type == DT.MR:
                return _MrXCatMeansSlice(self.dimensions, counts, base_counts)
            return _CatXCatMeansSlice(self.dimensions, counts, base_counts)

    @lazyproperty
    def _slice(self):
        """This is essentially a factory method.

        Needs to live (probably) in the _BaseSclice (which doesn't yet exist).
        It also needs to be tidied up a bit.
        """
        dimensions = self.dimensions
        base_counts = self._cube._apply_missings(
            # self._cube._measure(False).raw_cube_array
            self._cube._measures.unweighted_counts.raw_cube_array
        )
        counts_with_missings = self._cube._measure(self._weighted).raw_cube_array
        counts = self._cube._apply_missings(counts_with_missings)
        type_ = self._cube.dim_types[-2:]
        if self._cube.has_means:
            return self._create_means_slice(counts, base_counts)
        if self._cube.ndim > 2 or self._ca_as_0th:
            base_counts = base_counts[self._slice_idx]
            counts = counts[self._slice_idx]
            counts_with_missings = counts_with_missings[self._slice_idx]
            if self._cube.dim_types[0] == DT.MR:
                base_counts = base_counts[0]
                counts = counts[0]
                counts_with_missings = counts_with_missings[0]
            elif self._ca_as_0th:
                table_name = "%s: %s" % (
                    self._cube.dimensions[-2:][0].name,
                    self._cube.dimensions[-2:][0].valid_elements[self._slice_idx].label,
                )
                return _1DCaCatSlice(self.dimensions, counts, base_counts, table_name)
        elif self._cube.ndim < 2:
            if type_[0] == DT.MR:
                return _1DMrSlice(dimensions, counts, base_counts)
            return _1DCatSlice(dimensions, counts, base_counts)
        if type_ == (DT.MR, DT.MR):
            return _MrXMrSlice(dimensions, counts, base_counts, counts_with_missings)
        elif type_[0] == DT.MR:
            return _MrXCatSlice(dimensions, counts, base_counts, counts_with_missings)
        elif type_[1] == DT.MR:
            return _CatXMrSlice(dimensions, counts, base_counts, counts_with_missings)
        return _CatXCatSlice(dimensions, counts, base_counts, counts_with_missings)

    @lazyproperty
    def _transforms(self):
        return Transforms(self._slice, self.dimensions, self._pruning)


class _0DMeansSlice(object):
    """Represents slices with means (and no counts)."""

    # TODO: We might need to have 2 of these, one for 0-D, and one for 1-D mean cubes
    def __init__(self, means, base_counts):
        self._means = means
        self._base_counts = base_counts

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def table_margin(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def table_base(self):
        # TODO: Check why we expect mean instead of the real base in this case.
        return self.means


# Used to represent the non-existent dimension in case of 1D vectors (that need to be
# accessed as slices, to support cr.exporter).
_PlaceholderElement = namedtuple("_PlaceholderElement", "label, is_hidden")


class _1DMeansSlice(_0DMeansSlice):
    def __init__(self, dimension, means, base_counts):
        super(_1DMeansSlice, self).__init__(means, base_counts)
        self._dimension = dimension

    @lazyproperty
    def rows(self):
        """Rows for Means slice, that enable iteration over labels.

        These vectors are not used for any computations. `means` is used for that,
        directly. However, for the wirng of the exporter, these mean slices need to
        support some additional API, such as labels. And for that, they need to
        support row iteration.
        """
        return tuple(
            _MeansVector(element, base_counts, np.array([means]))
            for element, base_counts, means in zip(
                self._dimension.valid_elements, self._base_counts, self._means
            )
        )

    @lazyproperty
    def columns(self):
        """A single vector that is used only for pruning Means slices."""
        return (
            _BaseVector(_PlaceholderElement("Means Summary", False), self._base_counts),
        )

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)


class _1DMrWithMeansSlice(_1DMeansSlice):
    @lazyproperty
    def rows(self):
        return tuple(
            _MeansWithMrVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._dimension.valid_elements, self._base_counts, self._means
            )
        )


class _CatXCatSlice(object):
    """Deals with CAT x CAT data.

    Delegatest most functionality to vectors (rows or columns), but calculates some
    values by itself (like table_margin).

    This class (or its inheritants) must be instantiated as a starting point when
    dealing with slices. Other classes that represents various stages of
    transformations, need to repro a portion of this class' API (like iterating over
    rows or columns).
    """

    def __init__(self, dimensions, counts, base_counts, counts_with_missings=None):
        self._dimensions = dimensions
        self._counts = counts
        self._base_counts = base_counts
        self._all_counts = counts_with_missings

    @lazyproperty
    def _column_index(self):

        # TODO: This is a hack to make it work. It should be addressed properly with
        # passing `counts_with_missings` in all the right places in the factory.
        # Also - subclass for proper functionality in various MR cases.
        if self._all_counts is None:
            return self._column_proportions

        return self._column_proportions / self._baseline * 100

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For CAT x CAT slice it needs to
        sum across axis 1, because that's the columns CAT dimension, that gets
        collapsed. The total is calculated by totaling the unconditional row margin.
        Please note that the total _doesn't_ include missings for the row dimension.
        """
        dim_sum = np.sum(self._all_counts, axis=1)[self._valid_rows_idxs]
        return dim_sum[:, None] / np.sum(dim_sum)

    @lazyproperty
    def _valid_rows_idxs(self):
        """ndarray-style index for only valid rows (out of missing and not-missing)."""
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)

    @lazyproperty
    def _column_proportions(self):
        return np.array([col.proportions for col in self.columns]).T

    @lazyproperty
    def names(self):
        return tuple([dimension.name for dimension in self._dimensions])

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _row_elements(self):
        return self._row_dimension.valid_elements

    @lazyproperty
    def _column_elements(self):
        return self._column_dimension.valid_elements

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self._zscores,
            self._column_index,
        )

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T, self._base_counts.T, self._column_elements, self._zscores.T
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _CategoricalVector(
                counts, base_counts, element, self.table_margin, zscore, column_index
            )
            for (
                counts,
                base_counts,
                element,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def columns(self):
        return tuple(
            _CategoricalVector(counts, base_counts, element, self.table_margin, zscore)
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def _zscores(self):
        return self._scalar_type_std_res(
            self._counts,
            self.table_margin,
            np.sum(self._counts, axis=0),
            np.sum(self._counts, axis=1),
        )

    @staticmethod
    def _scalar_type_std_res(counts, total, colsum, rowsum):
        """Return ndarray containing standard residuals for category values.

        The shape of the return value is the same as that of *counts*.
        """
        expected_counts = expected_freq(counts)
        residuals = counts - expected_counts
        variance = (
            np.outer(rowsum, colsum)
            * np.outer(total - rowsum, total - colsum)
            / total ** 3
        )
        return residuals / np.sqrt(variance)


class _CatXCatMeansSlice(_CatXCatSlice):
    def __init__(self, dimensions, means, base_counts):
        super(_CatXCatMeansSlice, self).__init__(dimensions, None, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._row_dimension.valid_elements, self._base_counts, self._means
            )
        )

    @lazyproperty
    def columns(self):
        return tuple(
            _MeansVector(element, base_counts, means)
            for element, base_counts, means in zip(
                self._column_dimension.valid_elements,
                self._base_counts.T,
                self._means.T,
            )
        )


class _1DCatSlice(_CatXCatSlice):
    """Special case of CAT x CAT, where the 2nd CAT doesn't exist.

    Values are treated as rows, while there's only a single column (vector).
    """

    @lazyproperty
    def _zscores(self):
        # TODO: Fix with real zscores
        return tuple([np.nan for _ in self._counts])

    @lazyproperty
    def columns(self):
        return tuple(
            [
                _CategoricalVector(
                    self._counts,
                    self._base_counts,
                    _PlaceholderElement("Summary", False),
                    self.table_margin,
                )
            ]
        )


class _1DCaCatSlice(_1DCatSlice):
    def __init__(self, dimensions, counts, base_counts, table_name):
        super(_1DCaCatSlice, self).__init__(dimensions, counts, base_counts)
        self._table_name = table_name


class _SliceWithMR(_CatXCatSlice):
    @staticmethod
    def _array_type_std_res(counts, total, colsum, rowsum):
        expected_counts = rowsum * colsum / total
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)


class _MrXCatSlice(_SliceWithMR):
    """Represents MR x CAT slices.

    It's similar to CAT x CAT, other than the way it handles columns. For
    columns - which correspond to the MR dimension - it needs to handle the indexing
    of selected/not-selected correctly.
    """

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For MR x CAT slice it needs to
        sum across axis 2, because that's the CAT dimension, that gets collapsed.
        Then it needs to select only the selected counts of
        the MR (hence the `[:, 0]` index). The total needs include missings of the
        2nd dimension, but not of the first (hence the [:, 0:2] index, which only
        includes the selected and not-selected of the MR dimension).
        """
        dim_sum = np.sum(self._all_counts, axis=2)[:, 0][self._valid_rows_idxs]
        total = np.sum(self._all_counts[self._valid_rows_idxs][:, 0:2], axis=(1, 2))
        return (dim_sum / total)[:, None]

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, 0, :],
            self.table_margin[:, None],
            np.sum(self._counts, axis=1),
            np.sum(self._counts[:, 0, :], axis=1)[:, None],
        )

    @lazyproperty
    def rows(self):
        """Use only selected counts."""
        return tuple(
            _CatXMrVector(
                counts, base_counts, element, table_margin, zscore, column_index
            )
            for (
                counts,
                base_counts,
                element,
                table_margin,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._column_index,
        )

    @lazyproperty
    def columns(self):
        """Use bother selected and not-selected counts."""
        return tuple(
            _MultipleResponseVector(
                counts, base_counts, element, self.table_margin, zscore
            )
            for counts, base_counts, element, zscore in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 2))

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(1, 2))


class _MrXCatMeansSlice(_MrXCatSlice):
    def __init__(self, dimensions, means, base_counts):
        counts = np.empty(means.shape)
        super(_MrXCatMeansSlice, self).__init__(dimensions, counts, base_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        return tuple(
            _MeansWithMrVector(element, base_counts, means[0])
            for element, base_counts, means in zip(
                self._row_dimension.valid_elements, self._base_counts, self._means
            )
        )


class _1DMrSlice(_MrXCatSlice):
    """Special case of 1-D MR slice (vector)."""

    @lazyproperty
    def _zscores(self):
        return np.array([np.nan] * self._base_counts.shape[0])

    @lazyproperty
    def columns(self):
        return tuple(
            [
                _MultipleResponseVector(
                    self._counts.T,
                    self._base_counts.T,
                    _PlaceholderElement("Summary", False),
                    self.table_margin,
                )
            ]
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=1)


class _CatXMrSlice(_SliceWithMR):
    """Handles CAT x MR slices.

    Needs to handle correctly the indexing for the selected/not-selected for rows
    (which correspond to the MR dimension).
    """

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For CAT x MR slice it needs to sum
        across axis 2, because that's the MR_CAT dimension, that gets collapsed
        (but the MR subvars don't get collapsed). Then it needs to calculate the total,
        which is easily obtained by summing across the CAT dimension (hence `axis=0`).
        """
        dim_sum = np.sum(self._all_counts, axis=2)[self._valid_rows_idxs]
        return dim_sum / np.sum(dim_sum, axis=0)

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, :, 0],
            self.table_margin,
            np.sum(self._counts[:, :, 0], axis=0),
            np.sum(self._counts, axis=2),
        )

    @lazyproperty
    def rows(self):
        return tuple(
            _MultipleResponseVector(
                counts.T,
                base_counts.T,
                element,
                self.table_margin,
                zscore,
                column_index,
            )
            for (
                counts,
                base_counts,
                element,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def _column_generator(self):
        return zip(
            # self._counts.T[0],
            np.array([self._counts.T[0].T, self._counts.T[1].T]).T,
            # self._base_counts.T[0],
            np.array([self._base_counts.T[0].T, self._base_counts.T[1].T]).T,
            self._column_elements,
            self.table_margin,
        )

    @lazyproperty
    def columns(self):
        return tuple(
            # _CategoricalVector(counts, base_counts, element, table_margin)
            _CatXMrVector(counts.T, base_counts.T, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(0, 2))

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(0, 2))


class _MrXMrSlice(_SliceWithMR):
    """Represents MR x MR slices.

    Needs to properly index both rows and columns (selected/not-selected.
    """

    @lazyproperty
    def _zscores(self):
        return self._array_type_std_res(
            self._counts[:, 0, :, 0],
            self.table_margin,
            np.sum(self._counts, axis=1)[:, :, 0],
            np.sum(self._counts, axis=3)[:, 0, :],
        )

    @lazyproperty
    def _baseline(self):
        """ndarray of baseline values for column index.

        Baseline is obtained by calculating the unconditional row margin (which needs
        to include missings from the column dimension) and dividint it by the total.
        Total also needs to include the missings from the column dimension.

        `dim_sum` is the unconditional row margin. For MR x MR slice it needs to sum
        across axis 3, because that's the MR_CAT dimension, that gets collapsed
        (but the MR subvars don't get collapsed). Then it needs to calculate the total,
        which is obtained by summing across both MR_CAT dimensions. However, please
        note, that in calculating the unconditional total, missing elements need to be
        included for the column dimension, while they need to be _excluded_ for the row
        dimension. Hence the `[:, 0:2]` indexing for the first MR_CAT, but not the 2nd.
        """
        dim_sum = np.sum(self._all_counts[:, 0:2], axis=3)[self._valid_rows_idxs][:, 0]
        total = np.sum(self._all_counts[:, 0:2], axis=(1, 3))[self._valid_rows_idxs]
        return dim_sum / total

    @lazyproperty
    def _row_generator(self):
        return zip(
            self._counts,
            self._base_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._column_index,
        )

    @lazyproperty
    def rows(self):
        # return tuple(_MultipleResponseVector(counts[0].T) for counts in self._counts)
        return tuple(
            _MultipleResponseVector(
                counts[0].T,
                base_counts[0].T,
                element,
                table_margin,
                zscore,
                column_index,
            )
            for (
                counts,
                base_counts,
                element,
                table_margin,
                zscore,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def _column_generator(self):
        return zip(
            self._counts.T[0],
            self._base_counts.T[0],
            self._column_elements,
            self.table_margin.T,
        )

    @lazyproperty
    def columns(self):
        # return tuple(_MultipleResponseVector(counts) for counts in self._counts.T[0])
        return tuple(
            _MultipleResponseVector(counts, base_counts, element, table_margin)
            for counts, base_counts, element, table_margin in self._column_generator
        )

    @lazyproperty
    def table_margin(self):
        return np.sum(self._counts, axis=(1, 3))

    @lazyproperty
    def table_base(self):
        return np.sum(self._base_counts, axis=(1, 3))


class _BaseVector(object):
    def __init__(self, element, base_counts):
        self._element = element
        self._base_counts = base_counts

    @lazyproperty
    def is_insertion(self):
        return False

    @lazyproperty
    def numeric(self):
        return self._element.numeric_value

    @lazyproperty
    def label(self):
        return self._element.label

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts)

    @lazyproperty
    def pruned(self):
        return self.base == 0 or np.isnan(self.base)

    @lazyproperty
    def hidden(self):
        return self._element.is_hidden

    @lazyproperty
    def cat_id(self):
        return self._element.element_id


class _MeansVector(_BaseVector):
    def __init__(self, element, base_counts, means):
        super(_MeansVector, self).__init__(element, base_counts)
        self._means = means

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def values(self):
        return self._means


class _MeansWithMrVector(_MeansVector):
    """This is a row of a 1-D MR with Means.

    This vector is special in the sense that it doesn't provide us with the normal
    base (which is selected + not-selected for a 1-D MR _without_ means). Instead, it
    calculates the base as _just_ the selected, which is the correct base for
    the 1-D MR with means.
    """

    @lazyproperty
    def base(self):
        return np.sum(self._base_counts[0])


class _CategoricalVector(_BaseVector):
    """Main staple of all measures.

    Some of the measures it can calculate by itself, others it needs to receive at
    construction time (like table margin and zscores).
    """

    def __init__(
        self, counts, base_counts, element, table_margin, zscore=None, column_index=None
    ):
        super(_CategoricalVector, self).__init__(element, base_counts)
        self._counts = counts
        self._table_margin = table_margin
        self._zscore = zscore
        self._column_index = column_index

    @lazyproperty
    def column_index(self):
        return self._column_index

    @lazyproperty
    def pvals(self):
        return 2 * (1 - norm.cdf(np.abs(self._zscore)))

    @lazyproperty
    def zscore(self):
        return self._zscore

    @lazyproperty
    def values(self):
        if not isinstance(self._counts, np.ndarray):
            return np.array([self._counts])
        return self._counts

    @lazyproperty
    def base_values(self):
        if not isinstance(self._base_counts, np.ndarray):
            return np.array([self._base_counts])
        return self._base_counts

    @lazyproperty
    def margin(self):
        return np.sum(self._counts)

    @lazyproperty
    def table_margin(self):
        return self._table_margin

    @lazyproperty
    def proportions(self):
        return self.values / self.margin
        # return self.values / self.base


class _CatXMrVector(_CategoricalVector):
    def __init__(
        self, counts, base_counts, label, table_margin, zscore=None, column_index=None
    ):
        super(_CatXMrVector, self).__init__(
            counts[0], base_counts[0], label, table_margin, zscore, column_index
        )
        self._pruning_bases = base_counts

    @lazyproperty
    def pruned(self):
        return np.sum(self._pruning_bases) == 0


class _MultipleResponseVector(_CategoricalVector):
    """Handles MR vectors (either rows or columns)

    Needs to handle selected and not-selected properly. Consequently, it calculates
    the right margin (for itself), but receives table margin on construction
    time (from the slice).
    """

    @lazyproperty
    def values(self):
        return self._selected

    @lazyproperty
    def base_values(self):
        return self._base_counts[0, :]

    @lazyproperty
    def base(self):
        counts = zip(self._selected_unweighted, self._not_selected_unweighted)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def margin(self):
        counts = zip(self._selected, self._not_selected)
        return np.array(
            [selected + not_selected for (selected, not_selected) in counts]
        )

    @lazyproperty
    def pruned(self):
        return np.all(self.base == 0) or np.all(np.isnan(self.base))

    @lazyproperty
    def _selected(self):
        return self._counts[0, :]

    @lazyproperty
    def _not_selected(self):
        return self._counts[1, :]

    @lazyproperty
    def _selected_unweighted(self):
        return self._base_counts[0, :]

    @lazyproperty
    def _not_selected_unweighted(self):
        return self._base_counts[1, :]


class Insertions(object):
    """Represents slice's insertions (inserted rows and columns).

    It generates the inserted rows and columns directly from the slice, based on the
    subtotals.
    """

    def __init__(self, dimensions, slice_):
        self._dimensions = dimensions
        self._slice = slice_

    @lazyproperty
    def bottom_columns(self):
        return tuple(
            columns for columns in self._inserted_columns if columns.anchor == "bottom"
        )

    @lazyproperty
    def bottom_rows(self):
        return tuple(row for row in self._rows if row.anchor == "bottom")

    @lazyproperty
    def columns(self):
        return tuple(
            column
            for column in self._inserted_columns
            if column.anchor not in ("top", "bottom")
        )

    @lazyproperty
    def rows(self):
        return tuple(row for row in self._rows if row.anchor not in ("top", "bottom"))

    @lazyproperty
    def top_columns(self):
        return tuple(
            columns for columns in self._inserted_columns if columns.anchor == "top"
        )

    @lazyproperty
    def top_rows(self):
        return tuple(row for row in self._rows if row.anchor == "top")

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def _inserted_columns(self):
        """Sequence of _InsertionColumn objects representing subtotal columns."""
        # ---a 1D slice (strand) can have no inserted columns---
        if len(self._dimensions) < 2:
            return ()
        # ---an aggregate columns-dimension is not summable---
        if self._column_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        return tuple(
            _InsertionColumn(self._slice, subtotal)
            for subtotal in self._column_dimension.subtotals
        )

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _rows(self):
        if self._row_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        return tuple(
            _InsertionRow(self._slice, subtotal)
            for subtotal in self._row_dimension.subtotals
        )


class _InsertionVector(object):
    """Represents constituent vectors of the `Insertions` class.

    Needs to repro the API of the more basic vectors - because of
    composition (and not inheritance)
    """

    def __init__(self, slice_, subtotal):
        self._slice = slice_
        self._subtotal = subtotal

    @lazyproperty
    def is_insertion(self):
        return True

    @lazyproperty
    def means(self):
        return np.array([np.nan])

    @lazyproperty
    def cat_id(self):
        return -1

    @lazyproperty
    def numeric(self):
        return np.nan

    @lazyproperty
    def hidden(self):
        """Insertion cannot be hidden."""
        return False

    @lazyproperty
    def label(self):
        return self._subtotal.label

    @lazyproperty
    def table_margin(self):
        return self._slice.table_margin

    @lazyproperty
    def anchor(self):
        return self._subtotal.anchor_idx

    @lazyproperty
    def addend_idxs(self):
        return np.array(self._subtotal.addend_idxs)

    @lazyproperty
    def values(self):
        return np.sum(np.array([row.values for row in self._addend_vectors]), axis=0)

    @lazyproperty
    def base_values(self):
        return np.sum(
            np.array([row.base_values for row in self._addend_vectors]), axis=0
        )

    @lazyproperty
    def margin(self):
        return np.sum(np.array([vec.margin for vec in self._addend_vectors]), axis=0)

    @lazyproperty
    def base(self):
        return np.sum(np.array([vec.base for vec in self._addend_vectors]), axis=0)


class _InsertionColumn(_InsertionVector):
    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            column
            for i, column in enumerate(self._slice.columns)
            if i in self._subtotal.addend_idxs
        )

    @lazyproperty
    def pruned(self):
        return not np.any(np.array([row.base for row in self._slice.rows]))


class _InsertionRow(_InsertionVector):
    @lazyproperty
    def pvals(self):
        return np.array([np.nan] * len(self._slice.columns))

    @lazyproperty
    def zscore(self):
        return np.array([np.nan] * len(self._slice.columns))

    @lazyproperty
    def _addend_vectors(self):
        return tuple(
            row
            for i, row in enumerate(self._slice.rows)
            if i in self._subtotal.addend_idxs
        )

    @lazyproperty
    def pruned(self):
        return not np.any(np.array([column.base for column in self._slice.columns]))


class Assembler(object):
    """In charge of performing all the transforms sequentially."""

    def __init__(self, slice_, transforms):
        self._slice = slice_
        self._transforms = transforms

    @lazyproperty
    def slice(self):
        """Apply all transforms sequentially."""

        slice_ = OrderedSlice(self._slice, self._transforms)
        slice_ = SliceWithInsertions(slice_, self._transforms)
        slice_ = SliceWithHidden(slice_, self._transforms)
        slice_ = PrunedSlice(slice_, self._transforms)

        return slice_

    @lazyproperty
    def rows(self):
        return self.slice.rows

    @lazyproperty
    def columns(self):
        return self.slice.columns

    @lazyproperty
    def table_margin(self):
        return self.slice.table_margin

    @lazyproperty
    def table_margin_unpruned(self):
        return self.slice.table_margin_unpruned

    @lazyproperty
    def table_base(self):
        return self.slice.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self.slice.table_base_unpruned


class _TransformedSlice(object):
    def __init__(self, base_slice, transforms):
        self._base_slice = base_slice
        self._transforms = transforms

    @lazyproperty
    def table_margin(self):
        return self._base_slice.table_margin

    @lazyproperty
    def table_base(self):
        return self._base_slice.table_base


class SliceWithInsertions(_TransformedSlice):
    """Represents slice with both normal and inserted bits."""

    @lazyproperty
    def _insertions(self):
        return self._transforms.insertions

    @lazyproperty
    def rows(self):
        return tuple(self._top_rows + self._interleaved_rows + self._bottom_rows)

    @lazyproperty
    def columns(self):
        return tuple(
            self._top_columns + self._interleaved_columns + self._bottom_columns
        )

    @lazyproperty
    def _insertion_columns(self):
        return self._insertions._inserted_columns

    @lazyproperty
    def _insertion_rows(self):
        return self._insertions._rows

    @lazyproperty
    def _assembled_rows(self):
        return tuple(
            _AssembledVector(row, self._insertion_columns)
            for row in self._base_slice.rows
        )

    @lazyproperty
    def _bottom_rows(self):
        return tuple(
            _AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.bottom_rows
        )

    @lazyproperty
    def _top_rows(self):
        return tuple(
            _AssembledVector(row, self._insertion_columns)
            for row in self._insertions.top_rows
        )

    @lazyproperty
    def _top_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.top_columns
        )

    @lazyproperty
    def _bottom_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.bottom_columns
        )

    @lazyproperty
    def _assembled_columns(self):
        return tuple(
            _AssembledVector(column, self._insertion_rows)
            for column in self._base_slice.columns
        )

    @lazyproperty
    def _assembled_insertion_rows(self):
        return tuple(
            _AssembledInsertionVector(row, self._insertion_columns)
            for row in self._insertions.rows
        )

    @lazyproperty
    def _interleaved_rows(self):
        rows = []
        for i in range(len(self._base_slice.rows)):
            rows.append(self._assembled_rows[i])
            for insertion_row in self._assembled_insertion_rows:
                if i == insertion_row.anchor:
                    rows.append(insertion_row)
        return tuple(rows)

    @lazyproperty
    def _assembled_insertion_columns(self):
        return tuple(
            _AssembledInsertionVector(column, self._insertion_rows)
            for column in self._insertions.columns
        )

    @lazyproperty
    def _interleaved_columns(self):
        columns = []
        for i in range(len(self._base_slice.columns)):
            columns.append(self._assembled_columns[i])
            for insertion_column in self._assembled_insertion_columns:
                if i == insertion_column.anchor:
                    columns.append(insertion_column)
        return tuple(columns)


class _TransformedVecvtor(object):
    @lazyproperty
    def is_insertion(self):
        return self._base_vector.is_insertion

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def hidden(self):
        return self._base_vector.hidden

    @lazyproperty
    def pruned(self):
        return self._base_vector.pruned

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def numeric(self):
        return self._base_vector.numeric

    @lazyproperty
    def cat_id(self):
        return self._base_vector.cat_id

    @lazyproperty
    def means(self):
        return self._base_vector.means

    @lazyproperty
    def table_base(self):
        return self._base_vector.table_base

    @lazyproperty
    def table_margin(self):
        return self._base_vector.table_margin


class _AssembledVector(_TransformedVecvtor):
    """Vector with base, as well as inserted, elements (of the opposite dimension)."""

    def __init__(self, base_vector, opposite_inserted_vectors):
        self._base_vector = base_vector
        self._opposite_inserted_vectors = opposite_inserted_vectors

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def pvals(self):
        return (
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_pvals
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def zscore(self):
        return (
            tuple([np.nan] * len(self._top_values))
            + self._interleaved_zscore
            + tuple([np.nan] * len(self._bottom_values))
        )

    @lazyproperty
    def _interleaved_pvals(self):
        pvals = []
        for i, value in enumerate(self._base_vector.pvals):
            pvals.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    pvals.append(np.nan)
        return tuple(pvals)

    @lazyproperty
    def _interleaved_zscore(self):
        zscore = []
        for i, value in enumerate(self._base_vector.zscore):
            zscore.append(value)
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    zscore.append(np.nan)
        return tuple(zscore)

    @lazyproperty
    def margin(self):
        return self._base_vector.margin

    @lazyproperty
    def base(self):
        return self._base_vector.base

    @lazyproperty
    def pruned(self):
        return self._base_vector.pruned

    @lazyproperty
    def proportions(self):
        # return self.values / self.base
        return self.values / self.margin

    @lazyproperty
    def table_proportions(self):
        return self.values / self._base_vector.table_margin

    @lazyproperty
    def values(self):
        return np.array(
            self._top_values + self._interleaved_values + self._bottom_values
        )

    @lazyproperty
    def base_values(self):
        # TODO: Do for real
        return np.array(
            self._top_base_values
            + self._interleaved_base_values
            + self._bottom_base_values
        )

    @lazyproperty
    def _top_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _top_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "top"
        )

    @lazyproperty
    def _bottom_values(self):
        return tuple(
            np.sum(self._base_vector.values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _bottom_base_values(self):
        return tuple(
            np.sum(self._base_vector.base_values[col.addend_idxs])
            for col in self._opposite_inserted_vectors
            if col.anchor == "bottom"
        )

    @lazyproperty
    def _interleaved_values(self):
        values = []
        for i in range(len(self._base_vector.values)):
            values.append(self._base_vector.values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.values[inserted_vector.addend_idxs]
                    )
                    values.append(insertion_value)
        return tuple(values)

    @lazyproperty
    def _interleaved_base_values(self):
        base_values = []
        for i in range(len(self._base_vector.base_values)):
            base_values.append(self._base_vector.base_values[i])
            for inserted_vector in self._opposite_inserted_vectors:
                if i == inserted_vector.anchor:
                    insertion_value = np.sum(
                        self._base_vector.base_values[inserted_vector.addend_idxs]
                    )
                    base_values.append(insertion_value)
        return tuple(base_values)


class _AssembledInsertionVector(_AssembledVector):
    """Inserted row or col, but with elements from opposite dimension insertions.

    Needs to be subclassed from _AssembledVector, because it needs to provide the
    anchor, in order to know where it (itself) gets inserted.
    """

    @lazyproperty
    def anchor(self):
        return self._base_vector.anchor


# TODO: Not sure if Calculator is needed at all. It dupclicates most of the things
# from Assembler. Maybe just use one of those, and think of a better name.
class Calculator(object):
    """Calculates measures."""

    def __init__(self, assembler):
        self._assembler = assembler

    @lazyproperty
    def insertion_rows_idxs(self):
        return tuple(
            i for i, row in enumerate(self._assembler.rows) if row.is_insertion
        )

    @lazyproperty
    def insertion_columns_idxs(self):
        return tuple(
            i for i, column in enumerate(self._assembler.columns) if column.is_insertion
        )

    @lazyproperty
    def rows_dimension_numeric(self):
        return np.array([row.numeric for row in self._assembler.rows])

    @lazyproperty
    def columns_dimension_numeric(self):
        return np.array([column.numeric for column in self._assembler.columns])

    @lazyproperty
    def scale_means_row(self):
        if np.all(np.isnan(self.rows_dimension_numeric)):
            return None
        inner = np.nansum(self.rows_dimension_numeric[:, None] * self.counts, axis=0)
        not_a_nan_index = ~np.isnan(self.rows_dimension_numeric)
        denominator = np.sum(self.counts[not_a_nan_index, :], axis=0)
        return inner / denominator

    @lazyproperty
    def scale_means_row_margin(self):
        if np.all(np.isnan(self.rows_dimension_numeric)):
            return None

        row_margin = self.row_margin
        if len(row_margin.shape) > 1:
            # Hack for MR, where row margin is a table. Figure how to
            # fix with subclassing
            row_margin = row_margin[:, 0]

        not_a_nan_index = ~np.isnan(self.rows_dimension_numeric)
        return np.nansum(self.rows_dimension_numeric * row_margin) / np.sum(
            row_margin[not_a_nan_index]
        )

    @lazyproperty
    def scale_means_column_margin(self):
        if np.all(np.isnan(self.columns_dimension_numeric)):
            return None

        column_margin = self.column_margin
        if len(column_margin.shape) > 1:
            # Hack for MR, where column margin is a table. Figure how to
            # fix with subclassing
            column_margin = column_margin[0]

        not_a_nan_index = ~np.isnan(self.columns_dimension_numeric)
        return np.nansum(self.columns_dimension_numeric * column_margin) / np.sum(
            column_margin[not_a_nan_index]
        )

    @lazyproperty
    def scale_means_column(self):
        if np.all(np.isnan(self.columns_dimension_numeric)):
            return None

        inner = np.nansum(self.columns_dimension_numeric * self.counts, axis=1)
        not_a_nan_index = ~np.isnan(self.columns_dimension_numeric)
        denominator = np.sum(self.counts[:, not_a_nan_index], axis=1)
        return inner / denominator
        # return (
        #     np.nansum(self.columns_dimension_numeric * self.counts, axis=1)
        #     / self.row_margin
        # )

    @lazyproperty
    def column_index(self):
        return np.array([row.column_index for row in self._assembler.rows])

    @lazyproperty
    def pvals(self):
        return np.array([row.pvals for row in self._assembler.rows])

    @lazyproperty
    def row_proportions(self):
        return np.array([row.proportions for row in self._assembler.rows])

    @lazyproperty
    def column_proportions(self):
        return np.array([col.proportions for col in self._assembler.columns]).T

    @lazyproperty
    def table_proportions(self):
        return np.array([row.table_proportions for row in self._assembler.rows])

    @lazyproperty
    def row_margin(self):
        return np.array([row.margin for row in self._assembler.rows])

    @lazyproperty
    def column_margin(self):
        return np.array([column.margin for column in self._assembler.columns]).T

    @lazyproperty
    def table_margin(self):
        return self._assembler.table_margin

    @lazyproperty
    def row_base(self):
        return np.array([row.base for row in self._assembler.rows])

    @lazyproperty
    def column_base(self):
        return np.array([column.base for column in self._assembler.columns]).T

    @lazyproperty
    def table_base(self):
        return self._assembler.table_base

    @lazyproperty
    def table_base_unpruned(self):
        return self._assembler.table_base_unpruned

    @lazyproperty
    def table_margin_unpruned(self):
        return self._assembler.table_margin_unpruned

    @lazyproperty
    def counts(self):
        return np.array([row.values for row in self._assembler.rows])

    @lazyproperty
    def means(self):
        if type(self._assembler._slice) is _0DMeansSlice:
            return self._assembler._slice.means
        return np.array([row.means for row in self._assembler.rows])

    @lazyproperty
    def base_counts(self):
        return np.array([row.base_values for row in self._assembler.rows])

    @lazyproperty
    def row_labels(self):
        return tuple(row.label for row in self._assembler.rows)

    # TODO: Purge this once we do the transforms properly. It's only needed because of
    # old-style transforms in exporter
    @lazyproperty
    def column_labels_with_ids(self):
        return tuple(
            (column.label, column.cat_id) for column in self._assembler.columns
        )

    @lazyproperty
    def column_labels(self):
        return tuple(column.label for column in self._assembler.columns)

    @lazyproperty
    def zscore(self):
        return np.array([row.zscore for row in self._assembler.rows])


class OrderTransform(object):
    """Creates ordering indexes for rows and columns based on element ids."""

    def __init__(self, dimensions):
        self._dimensions = dimensions

    @lazyproperty
    def _row_dimension(self):
        return self._dimensions[0]

    @lazyproperty
    def _column_dimension(self):
        return self._dimensions[1]

    @lazyproperty
    def row_order(self):
        return np.array(self._row_dimension.valid_display_order)

    @lazyproperty
    def column_order(self):
        # If there's no column dimension, there can be no reordering for it
        if len(self._dimensions) < 2:
            return slice(None)
        return np.array(self._column_dimension.valid_display_order)


class OrderedVector(_TransformedVecvtor):
    """In charge of indexing elements properly, after ordering transform."""

    def __init__(self, vector, order):
        self._base_vector = vector
        self._order = order

    @lazyproperty
    def column_index(self):
        return self._base_vector.column_index

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def order(self):
        return self._order if self._order is not None else slice(None)

    @lazyproperty
    def values(self):
        return self._base_vector.values[self.order]

    @lazyproperty
    def base_values(self):
        return self._base_vector.base_values[self.order]

    @lazyproperty
    def zscore(self):
        return self._base_vector.zscore

    @lazyproperty
    def pvals(self):
        return self._base_vector.pvals

    @lazyproperty
    def base(self):
        return self._base_vector.base


class OrderedSlice(_TransformedSlice):
    """Result of the ordering transform.

    In charge of indexing rows and columns properly.
    """

    @lazyproperty
    def _ordering(self):
        return self._transforms.ordering

    @lazyproperty
    def rows(self):
        return tuple(
            OrderedVector(row, self._ordering.column_order)
            for row in tuple(np.array(self._base_slice.rows)[self._ordering.row_order])
        )

    @lazyproperty
    def columns(self):
        return tuple(
            OrderedVector(column, self._ordering.row_order)
            for column in tuple(
                np.array(self._base_slice.columns)[self._ordering.column_order]
            )
        )


class Transforms(object):
    """Container for the transforms."""

    def __init__(self, slice_, dimensions, pruning=None):
        self._slice = slice_
        self._dimensions = dimensions
        self._pruning = pruning

    @lazyproperty
    def ordering(self):
        return OrderTransform(self._dimensions)

    @lazyproperty
    def pruning(self):
        return self._pruning

    @lazyproperty
    def insertions(self):
        return Insertions(self._dimensions, self._slice)


class PrunedVector(_TransformedVecvtor):
    """Vector with elements from the opposide dimensions pruned."""

    def __init__(self, base_vector, opposite_vectors):
        self._base_vector = base_vector
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def means(self):
        return np.array(
            [
                means
                for means, opposite_vector in zip(
                    self._base_vector.means, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def column_index(self):
        return np.array(
            [
                column_index
                for column_index, opposite_vector in zip(
                    self._base_vector.column_index, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def zscore(self):
        return np.array(
            [
                zscore
                for zscore, opposite_vector in zip(
                    self._base_vector.zscore, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def label(self):
        return self._base_vector.label

    @lazyproperty
    def pvals(self):
        return np.array(
            [
                pvals
                for pvals, opposite_vector in zip(
                    self._base_vector.pvals, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def values(self):
        return np.array(
            [
                value
                for value, opposite_vector in zip(
                    self._base_vector.values, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def base_values(self):
        return np.array(
            [
                value
                for value, opposite_vector in zip(
                    self._base_vector.base_values, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def proportions(self):
        return np.array(
            [
                proportion
                for proportion, opposite_vector in zip(
                    self._base_vector.proportions, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def table_proportions(self):
        return np.array(
            [
                proportion
                for proportion, opposite_vector in zip(
                    self._base_vector.table_proportions, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def margin(self):
        if not isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin
        return np.array(
            [
                margin
                for margin, opposite_vector in zip(
                    self._base_vector.margin, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )

    @lazyproperty
    def base(self):
        if not isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base
        return np.array(
            [
                base
                for base, opposite_vector in zip(
                    self._base_vector.base, self._opposite_vectors
                )
                if not opposite_vector.pruned
            ]
        )


class PrunedSlice(_TransformedSlice):
    """Slice with rows or columns pruned.

    While the rows and/or columns need to be pruned, each one of the remaining
    vectors also needs to be pruned based on the opposite dimension's base.
    """

    @lazyproperty
    def _applied(self):
        return self._transforms._pruning

    @lazyproperty
    def rows(self):
        if not self._applied:
            return self._base_slice.rows

        return tuple(
            PrunedVector(row, self._base_slice.columns)
            for row in self._base_slice.rows
            if not row.pruned
        )

    @lazyproperty
    def columns(self):
        if not self._applied:
            return self._base_slice.columns

        return tuple(
            PrunedVector(column, self._base_slice.rows)
            for column in self._base_slice.columns
            if not column.pruned
        )

    @lazyproperty
    def table_margin_unpruned(self):
        return self._base_slice.table_margin

    @lazyproperty
    def table_base_unpruned(self):
        return self._base_slice.table_base

    @lazyproperty
    def table_margin(self):
        if not self._applied:
            return self._base_slice.table_margin

        margin = self._base_slice.table_margin
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]

    @lazyproperty
    def table_base(self):
        if not self._applied:
            return self._base_slice.table_base

        margin = self._base_slice.table_base
        index = margin != 0
        if margin.ndim < 2:
            return margin[index]
        row_ind = np.any(index, axis=1)
        col_ind = np.any(index, axis=0)
        return margin[np.ix_(row_ind, col_ind)]


class SliceWithHidden(_TransformedSlice):
    @lazyproperty
    def rows(self):
        return tuple(row for row in self._base_slice.rows if not row.hidden)

    @lazyproperty
    def columns(self):
        return tuple(
            HiddenVector(column, self._base_slice.rows)
            for column in self._base_slice.columns
            if not column.hidden
        )


class HiddenVector(_TransformedVecvtor):
    def __init__(self, base_vector, opposite_vectors):
        self._base_vector = base_vector
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def proportions(self):
        return np.array(
            [
                proportion
                for proportion, opposite_vector in zip(
                    self._base_vector.proportions, self._opposite_vectors
                )
                if not opposite_vector.hidden
            ]
        )
