# encoding: utf-8

"""A matrix is the 2D cube-data partition used by a slice.

A matrix object has rows and columns.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import itertools
import sys

import numpy as np
from scipy.stats import norm
from scipy.stats.contingency import expected_freq

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty, calculate_overlap_tstats


class TransformedMatrix(object):
    """Matrix reflecting application of ordering, insertion, and hiding transforms."""

    def __init__(self, unordered_matrix):
        self._unordered_matrix = unordered_matrix

    @classmethod
    def matrix(cls, cube, dimensions, slice_idx):
        """-> TransformedMatrix object constructed from values of `cube`.

        `cube` is the `cr.cube.Cube` object containing the data for this matrix. Note
        that not all the data in `cube` will necessarily be used by this matrix. When
        `cube` is more than 2-dimensional, it is "sliced" and each slice gets its own
        matrix (and `_Slice` object).

        `dimensions` is a pair (2-tuple) of (rows_dimension, columns_dimension)
        Dimension objects. These are always the last two dimensions of `cube` but may
        and often do have transformations applied that are not present on the `cube`
        dimensions from which they derive.

        `slice_idx` is an int offset indicating which portion of `cube` data to use for
        this matrix. There is one slice for each element of the first cube dimension
        (the "table" dimension) when the cube has more than two dimensions.
        """
        return cls(_BaseBaseMatrix.factory(cube, dimensions, slice_idx))

    @lazyproperty
    def columns(self):
        """A sequence of `_VectorAfterHiding` objects each representing a matrix column.

        Each column has all transformations applied, in particular, renaming, ordering,
        insertions (subtotals) and hiding (including pruning) of elements. The value in
        any given "cell" of each column also appears in a "cell" of its opposing row
        vector. Accessing values row-wise or column-wise produces the same overall
        matrix of values, although most commonly values are accessed row-wise.
        """
        return tuple(
            _VectorAfterHiding(column, self._assembled_rows)
            for column in self._assembled_columns
            if not column.hidden
        )

    @lazyproperty
    def overlaps_tstats(self):
        return (
            self._unordered_matrix.overlaps_tstats
            if self._unordered_matrix.is_augmented
            else None
        )

    @lazyproperty
    def rows(self):
        """A sequence of `_VectorAfterHiding` objects each representing a matrix row.

        Each row has all transformations applied, in particular, renaming, ordering,
        insertions (subtotals) and hiding (including pruning) of elements. The value in
        any given "cell" of each row also appears in a "cell" of its opposing column
        vector. Accessing values row-wise or column-wise produces the same overall
        matrix of values, although most commonly values are accessed row-wise.
        """
        return tuple(
            _VectorAfterHiding(row, self._assembled_columns)
            for row in self._assembled_rows
            if not row.hidden
        )

    @lazyproperty
    def table_base(self):
        """np.int64 scalar or a 1D or 2D ndarray of np.int64 representing table base.

        A multiple-response (MR) dimension produces an array of table-base values
        because each element (subvariable) of the dimension represents a logically
        distinct question which may not have been asked of all respondents. When both
        dimensions are MR, the return value is a 2D ndarray and there is a distinct
        table-base value for each "cell" of the matrix. A CAT_X_CAT matrix produces
        a scalar value for this property.
        """
        return self.table_base_unpruned[
            np.ix_(self._visible_rows_mask, self._visible_cols_mask)
        ]

    @lazyproperty
    def table_base_unpruned(self):
        """np.int64 scalar or a 1D or 2D ndarray of np.int64 representing table base.

        This version includes hidden vectors, those with either a hide transform on
        their element or that have been pruned (because their base (N) is zero). This
        does not affect a scalar value but when the return value is an ndarray, the
        shape may be different than the array returned by `.table_base`.

        A multiple-response (MR) dimension produces an array of table-base values
        because each element (subvariable) of the dimension represents a logically
        distinct question which may not have been asked of all respondents. When both
        dimensions are MR, the return value is a 2D ndarray. A CAT_X_CAT matrix produces
        a scalar value for this property.
        """
        # TODO: This the name is misleading. It's not only "unpruned" it's
        # "before_hiding" (of either kind, prune or hide-transform). But the real
        # problem is having this interface property at all. The need for this is related
        # to expressing ranges for base and margin in cubes that have an MR dimension.
        # The real solution is to compute ranges in `cr.cube` rather than leaking this
        # sort of internal detail through the interface and making the client compute
        # those for themselves. So this will require reconstructing that "show-ranges"
        # requirement and either adding some sort of a `.range` property that returns
        # a sequence of (min, max) tuples, or maybe just returning margin or base as
        # tuples when appropriate and having something like a `.margin_is_ranges`
        # predicate the client can switch on to control their rendering.
        return self._unordered_matrix.table_base

    @lazyproperty
    def table_margin(self):
        """np.float/int64 scalar or a 1D or 2D np.float/int64 ndarray table margin.

        The table margin is the overall sample size of the matrix. This is the weighted
        count of respondents who were asked both the rows-question *and* the
        columns-questions and provided a valid response for both (note that
        not-selecting an MR option/subvar is a valid response).

        A multiple-response (MR) dimension produces an array of table-margin values
        because each element (subvariable) of the dimension represents a logically
        distinct question which may not have been asked of all respondents. When both
        dimensions are MR, the return value is a 2D ndarray and there is a distinct
        table-margin value for each "cell" of the matrix. A CAT_X_CAT matrix produces
        a scalar value for this property.
        """
        return self.table_margin_unpruned[
            np.ix_(self._visible_rows_mask, self._visible_cols_mask)
        ]

    @lazyproperty
    def table_margin_unpruned(self):
        """np.float/int64 scalar or a 1D or 2D ndarray of np.float/int64 table margin.

        This version includes hidden vectors, those with either a hide transform on
        their element or that have been pruned (because their base (N) is zero). This
        does not affect a scalar value but when the return value is an ndarray, the
        shape may be different than the array returned by `.table_margin`.

        A matrix with a multiple-response (MR) dimension produces an array of
        table-margin values because each element (subvariable) of the dimension
        represents a logically distinct question which may not have been asked of all
        respondents. When both dimensions are MR, the return value is a 2D ndarray.
        A CAT_X_CAT matrix produces a scalar value for this property.
        """
        # TODO: see TODO in `.table_base_unpruned`
        return self._unordered_matrix.table_margin

    @lazyproperty
    def _assembled_columns(self):
        """Sequence of column vectors including inserted columns.

        Each column vector also includes any new elements introduced by inserted rows.

        Columns (_AssembledVector objects) appear in display-order with inserted columns
        appearing in the proper position relative to the base columns. Note the final
        appearance and absolute position of columns is subject to later column hiding.
        """
        return self._assembled_vectors(
            self._base_columns, self._inserted_columns, self._inserted_rows
        )

    @lazyproperty
    def _assembled_rows(self):
        """Sequence of row vectors including inserted rows.

        Each row vector also reflects any new elements introduced by inserted columns.
        """
        return self._assembled_vectors(
            self._base_rows, self._inserted_rows, self._inserted_columns
        )

    def _assembled_vectors(
        self, base_vectors, inserted_vectors, opposing_inserted_vectors
    ):
        """Sequence of vectors (rows or columns) including inserted vectors.

        Each opposing vector also includes any new elements introduced by insertions.

        The returned _AssembledVector objects appear in display-order with inserted
        vectors appearing in the proper position relative to the base vectors. Note the
        final appearance and absolute position of vectors is subject to later vector
        hiding.

        Vector ordering is accomplished by *sorting* the vectors on their `.ordering`
        value. The ordering value is a `(position, index, self)` triple.

        The int position value is roughly equivalent to the notion of "anchor". It is
        0 for anchor=="top", sys.maxsize for anchor=="bottom", and int(anchor_idx) + 1
        otherwise. The +1 ensures inserted vectors appear *after* the vector they are
        anchored to.

        The `index` value is the *negative* index of this subtotal in its collection
        (i.e. the "distance from the end" of this inserted vector). This ensures that an
        inserted vector will always sort *prior* to a base vector with the same position
        while preserving the payload order of the inserted vector when two or more are
        anchored to the same base vector.

        For a base-vector, `position` and `index` are the same.
        """
        return tuple(
            _AssembledVector(vector, opposing_inserted_vectors, 0 if idx < 0 else idx)
            for _, idx, vector in sorted(
                itertools.chain(
                    (bv.ordering for bv in base_vectors),
                    (iv.ordering for iv in inserted_vectors),
                )
            )
        )

    @lazyproperty
    def _base_columns(self):
        """Sequence of column vectors after ordering but prior to insertions."""
        return tuple(
            _OrderedVector(column, self._row_order, idx)
            for idx, column in enumerate(
                tuple(np.array(self._unordered_matrix.columns)[self._column_order])
            )
        )

    @lazyproperty
    def _base_rows(self):
        """Sequence of row vectors, after ordering but prior to insertions."""
        return tuple(
            _OrderedVector(row, self._column_order, idx)
            for idx, row in enumerate(
                tuple(np.array(self._unordered_matrix.rows)[self._row_order])
            )
        )

    @lazyproperty
    def _column_order(self):
        """ -> 1D ndarray of int col idx specifying order of unordered-array columns."""
        # --- Specifying int type prevents failure when there are zero columns. The
        # --- default type for ndarray is float, which is not valid for indexing.
        return np.array(self._columns_dimension.display_order, dtype=int)

    @lazyproperty
    def _columns_dimension(self):
        """The `Dimension` object representing column elements in this matrix."""
        return self._unordered_matrix.columns_dimension

    @lazyproperty
    def _inserted_columns(self):
        """-> tuple of _InsertedColumn objects representing subtotal columns.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order. All subtotals defined on the column
        dimension appear in the sequence.
        """
        # --- an aggregate columns-dimension is not summable ---
        if self._columns_dimension.dimension_type in (DT.MR, DT.CA):
            return ()

        # --- inserted vectors are indexed using their *negative* idx, i.e. their
        # --- "distance" from the end of the subtotals sequence. This insures their
        # --- ordering tuple sorts before all base-columns with the same position while
        # --- still providing an idx that works for indexed access (if required).
        subtotals = self._columns_dimension.subtotals
        neg_idxs = range(-len(subtotals), 0)  # ---like [-3, -2, -1]---

        return tuple(
            _InsertedColumn(
                subtotal,
                neg_idx,
                self._unordered_matrix.table_margin,
                self._base_rows,
                self._base_columns,
            )
            for subtotal, neg_idx in zip(subtotals, neg_idxs)
        )

    @lazyproperty
    def _inserted_rows(self):
        """-> tuple of _InsertedRow objects representing inserted subtotal rows.

        The returned vectors are in the order subtotals were specified in the cube
        result, which is no particular order.
        """
        # --- an aggregate rows-dimension is not summable ---
        if self._rows_dimension.dimension_type in (DT.MR, DT.CA):
            return tuple()

        subtotals = self._rows_dimension.subtotals
        neg_idxs = range(-len(subtotals), 0)  # ---like [-3, -2, -1]---

        return tuple(
            _InsertedRow(
                subtotal,
                neg_idx,
                self._unordered_matrix.table_margin,
                self._base_rows,
                self._base_columns,
            )
            for subtotal, neg_idx in zip(subtotals, neg_idxs)
        )

    @lazyproperty
    def _row_order(self):
        """ -> 1D ndarray of int row idx specifying order of unordered-array rows."""
        # --- specifying int type prevents failure when there are zero rows ---
        return np.array(self._rows_dimension.display_order, dtype=int)

    @lazyproperty
    def _rows_dimension(self):
        """The `Dimension` object representing row elements in this matrix."""
        return self._unordered_matrix.rows_dimension

    @lazyproperty
    def _visible_cols_mask(self):
        """Sequence of bool indicating which columns are visible (not hidden).

        Suitable for use as a numpy mask on the columns dimension.
        """
        return tuple(not col.hidden for col in self._assembled_columns)

    @lazyproperty
    def _visible_rows_mask(self):
        """Sequence of bool indicating which rows are visible (not hidden).

        Suitable for use as a numpy mask on the rows dimension.
        """
        return tuple(not row.hidden for row in self._assembled_rows)


# === BASE-MATRIX OBJECTS ===


class _BaseBaseMatrix(object):
    """Base class for all matrix (2D secondary-analyzer) objects."""

    def __init__(self, dimensions, counts, unweighted_counts):
        self._dimensions = dimensions
        self._counts = counts
        self._unweighted_counts = unweighted_counts

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return a base-matrix object of appropriate type for `cube`."""
        if cube.is_mr_aug:
            return cls._mr_aug_matrix_factory(cube, dimensions, slice_idx)

        # --- means cube gets one of the means-matrix types ---
        if cube.has_means:
            return cls._means_matrix_factory(cube, dimensions, slice_idx)

        # --- everything else gets a more conventional matrix ---
        return cls._regular_matrix_factory(cube, dimensions, slice_idx)

    @lazyproperty
    def columns_dimension(self):
        """The `Dimension` object representing column elements of this matrix."""
        return self._dimensions[1]

    @lazyproperty
    def rows_dimension(self):
        """The `Dimension` object representing row elements of this matrix."""
        return self._dimensions[0]

    @lazyproperty
    def table_margin(self):
        """np.float/int64 scalar or a 1D or 2D np.float/int64 ndarray table margin.

        The table margin is the overall sample size of the matrix. This is the weighted
        count of respondents who were asked both questions and provided a valid response
        for both (including not-selecting an MR option/subvar).

        A matrix with a multiple-response (MR) dimension produces a 1D ndarray value.
        When both dimensions are MR, the return value is a 2D ndarray and there is
        a distinct table-base value for each "cell" of the matrix. A CAT_X_CAT matrix
        produces a scalar value for this property.
        """
        raise NotImplementedError(
            "must be implemented by each subclass"
        )  # pragma: no cover

    def _array_type_std_res(self, counts, total, colsum, rowsum):
        """-> 2D ndarray of np.float64 std-res value for each cell of MR matrix.

        This is a utility method used by a matrix with one or more MR dimensions. The
        caller forms the input arrays based on which of its dimensions are MR.
        """
        # --- if the matrix is "defective", in the sense that it doesn't have at least
        # --- two rows and two columns that are "full" of data, don't calculate zscores.
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            return np.zeros(counts.shape)

        expected_counts = rowsum * colsum / total
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)

    @lazyproperty
    def _column_elements(self):
        """Sequence of `cr.cube.dimension._Element` object for each matrix column."""
        return self.columns_dimension.valid_elements

    @property
    def _column_generator(self):
        """Iterable providing construction parameters for each column vector in turn.

        Used by `.columns` property in each subclass. Cannot be a lazyproperty because
        an iterator is exhausted on each use.
        """
        # --- note zip() returns an iterator in Python 3 ---
        return zip(
            self._counts.T,
            self._unweighted_counts.T,
            self._column_elements,
            self._zscores.T,
            self._table_std_dev.T,
            self._table_std_err.T,
        )

    @lazyproperty
    def _column_proportions(self):
        """2D ndarray of np.float64 between 0.0 and 1.0.

        The value represents the ratio of each cell count to the total count (margin)
        for its column.
        """
        return np.array([col.proportions for col in self.columns]).T

    @classmethod
    def _means_matrix_factory(cls, cube, dimensions, slice_idx):
        """ -> matrix object appropriate to means `cube`."""
        dimension_types = cube.dimension_types[-2:]

        if dimension_types == (DT.MR, DT.MR):
            # --- this MEANS_MR_X_MR case hasn't arisen yet ---
            raise NotImplementedError(
                "MR x MR with means is not implemented."
            )  # pragma: no cover

        MatrixCls = (
            _MrXCatMeansMatrix
            if dimension_types[0] == DT.MR
            else _CatXMrMeansMatrix
            if dimension_types[1] == DT.MR
            else _CatXCatMeansMatrix
        )
        counts, unweighted_counts = (
            (cube.counts[slice_idx], cube.unweighted_counts[slice_idx])
            if cube.ndim == 3
            else (cube.counts, cube.unweighted_counts)
        )
        return MatrixCls(dimensions, counts, unweighted_counts)

    @classmethod
    def _mr_aug_matrix_factory(cls, cube, dimensions, slice_idx):
        """ -> matrix for MR_AUG slice."""
        overlap_tstats = calculate_overlap_tstats(
            _MrXMrMatrix,
            offset=1 if cube.dimension_types[0] == DT.MR else 0,
            mr_dimensions=dimensions,
            mr_counts=cube.counts,
            mr_unweighted_counts=cube.unweighted_counts,
            mr_counts_with_missings=cube.counts_with_missings,
            subtotals=cube.dimensions[0].subtotals,
        )

        # --- drop repeated MR dimension for matrix purposes ---
        dimensions = cube.dimensions[:-1]

        if cube.dimension_types[0] == DT.MR:
            return _MrXMrMatrix(
                dimensions,
                counts=np.sum(cube.counts[:, :, :, :, 0], axis=4),
                unweighted_counts=np.sum(cube.unweighted_counts[:, :, :, :, 0], axis=4),
                counts_with_missings=np.sum(
                    cube.counts_with_missings[:, :, :, :, 0], axis=4
                ),
                overlaps=overlap_tstats,
            )

        # --- otherwise rows-dimension is CAT ---
        return _CatXMrMatrix(
            dimensions,
            counts=np.sum(cube.counts[:, :, :, 0], axis=3),
            unweighted_counts=np.sum(cube.unweighted_counts[:, :, :, 0], axis=3),
            counts_with_missings=np.sum(cube.counts_with_missings[:, :, :, 0], axis=3),
            overlaps=overlap_tstats,
        )

    @classmethod
    def _regular_matrix_factory(cls, cube, dimensions, slice_idx):
        """ -> matrix object for non-mr-aug and non-means slice."""
        counts, unweighted_counts, counts_with_missings = (
            (
                cube.counts[slice_idx][0],
                cube.unweighted_counts[slice_idx][0],
                cube.counts_with_missings[slice_idx][0],
            )
            if cube.ndim > 2 and cube.dimension_types[0] == DT.MR
            else (
                cube.counts[slice_idx],
                cube.unweighted_counts[slice_idx],
                cube.counts_with_missings[slice_idx],
            )
            if cube.ndim > 2
            else (cube.counts, cube.unweighted_counts, cube.counts_with_missings)
        )

        dimension_types = cube.dimension_types[-2:]
        if dimension_types == (DT.MR, DT.MR):
            return _MrXMrMatrix(
                dimensions, counts, unweighted_counts, counts_with_missings
            )
        if dimension_types[0] == DT.MR:
            return _MrXCatMatrix(
                dimensions, counts, unweighted_counts, counts_with_missings
            )
        if dimension_types[1] == DT.MR:
            return _CatXMrMatrix(
                dimensions, counts, unweighted_counts, counts_with_missings
            )
        return _CatXCatMatrix(
            dimensions, counts, unweighted_counts, counts_with_missings
        )

    @lazyproperty
    def _row_elements(self):
        """Sequence of `cr.cube.dimension._Element` object for each matrix row."""
        return self.rows_dimension.valid_elements

    @lazyproperty
    def _valid_row_idxs(self):
        """ndarray-style index for only valid (non-missing) rows.

        Suitable for indexing a raw measure array to include only valid rows.
        """
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)


class _CatXCatMatrix(_BaseBaseMatrix):
    """Matrix for CAT_X_CAT cubes and base class for most other matrix classes.

    Delegates most functionality to vectors (rows or columns), but calculates some
    values by itself (like table_margin).

    `counts_with_missings` is the raw weighted counts array, needed to compute the
    column-index.
    """

    def __init__(
        self, dimensions, counts, unweighted_counts, counts_with_missings=None
    ):
        super(_CatXCatMatrix, self).__init__(dimensions, counts, unweighted_counts)
        self._counts_with_missings = counts_with_missings

    @lazyproperty
    def columns(self):
        """Sequence of `_CategoricalVector` object for each valid column element."""
        return tuple(
            _CategoricalVector(
                counts,
                unweighted_counts,
                element,
                self.table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                opposing_margin=self._rows_margin,
            )
            for (
                counts,
                unweighted_counts,
                element,
                zscores,
                table_std_dev,
                table_std_err,
            ) in self._column_generator
        )

    @lazyproperty
    def rows(self):
        """Sequence of `_CategoricalVector` object for each valid row element."""
        return tuple(
            _CategoricalVector(
                counts,
                unweighted_counts,
                element,
                self.table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
                opposing_margin=self._columns_margin,
            )
            for (
                counts,
                unweighted_counts,
                element,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        """np.int64 count of actual respondents who answered both questions.

        Each dimension of a CAT_X_CAT matrix represents a categorical question. Only
        responses that include answers to both those questions appear as entries in the
        valid elements of those dimensions. The sum total of all valid answers is the
        sample size, aka "N" or "base".
        """
        return np.sum(self._unweighted_counts)

    @lazyproperty
    def table_margin(self):
        """Scalar np.float/int64 weighted-N for overall table.

        This is the weighted count of respondents who provided a valid response to
        both questions. Because both dimensions are CAT, the table-margin value is the
        same for all cells of the matrix. Value is np.int64 when source cube-result is
        unweighted.
        """
        return np.sum(self._counts)

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each row in matrix.

        The shape of the return value is (nrows, 1). The baseline value for a row is the
        proportion of all values that appear in that row. A baseline for a 4 x 3 matrix
        looks like this:

            [[0.2006734 ]
             [0.72592593]
             [0.05521886]
             [0.01818182]]

        Note that the baseline values sum to 1.0. This is because each represents the
        portion of all responses that fall in that row. This baseline value is the
        denominator of the `._column_index` computation.

        Baseline is a straightforward function of the *unconditional row margin*.
        Unconditional here means that both valid and invalid responses (to the
        columns-var question) are included. This ensures that the baseline is not
        distorted by a large number of missing responses to the columns-question.
        """
        # --- uncond_row_margin is a 1D ndarray of the weighted total observation count
        # --- involving each valid row. Counts consider both valid and invalid columns,
        # --- but are only produced for valid rows.
        uncond_row_margin = np.sum(self._counts_with_missings, axis=1)[
            self._valid_row_idxs
        ]
        return uncond_row_margin[:, None] / np.sum(uncond_row_margin)

    @lazyproperty
    def _column_index(self):
        """2D np.float64/np.nan ndarray of column-index value for each matrix cell.

        Column-index answers the question "are respondents in this row-category more or
        less likely than the overall table population to choose the answer represented
        by this column?". For example, if the row is "Hispanic" and the column is
        home-ownership, a value of 100 indicates hispanics are no less and no more
        likely to own their home than the overall population. If that value was 150, it
        would indicate hispanics are 50% more likely to own their home than the general
        population (or the population surveyed anyway).
        These values must be known by vectors but cannot be calculated there (the
        calculation is "cross-vector") so these values must be passed down from the
        matrix level.
        """
        return self._column_proportions / self._baseline * 100

    @lazyproperty
    def _columns_margin(self):
        """1D ndarray of np.float64 (or np.int64) weighted N for each matrix column.

        These values are required for zscore calculations at vector level but must be
        calculated at matrix level. The values are `np.int64` when the source
        cube-result is not weighted.
        """
        return np.sum(self._counts, axis=0)

    @property
    def _row_generator(self):
        """Iterable of arguments to row-vector constructor call for each row element.

        Used by `.rows` property. Cannot be a lazyproperty because an iterator is
        exhausted on each use and must be created newly on each call.
        """
        # --- Note `zip()` returns an iterator in Python 3 ---
        return zip(
            self._counts,
            self._unweighted_counts,
            self._row_elements,
            self._zscores,
            self._table_std_dev,
            self._table_std_err,
            self._column_index,
        )

    @lazyproperty
    def _rows_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each matrix row.

        These "opposing-margin" values are required for vector-level calculations but
        cannot be calculated there (the calculation is "cross-vector") so these values
        must be passed down from the matrix level. The values are `np.int64` when the
        source cube-result is not weighted.
        """
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def _table_proportion_variance(self):
        """2D ndarray of np.float64 cell proportion variance for each cell of matrix."""
        p = self._counts / self.table_margin
        return p * (1 - p)

    @lazyproperty
    def _table_std_dev(self):
        """2D np.float64 ndarray of table-percent std-deviation for each matrix cell.

        Standard deviation is the square-root of the variance.
        """
        return np.sqrt(self._table_proportion_variance)

    @lazyproperty
    def _table_std_err(self):
        """2D np.float64 ndarray of table-percent std-error for each matrix cell.

        Standard error is sqrt(variance/N).
        """
        return np.sqrt(self._table_proportion_variance / self.table_margin)

    @lazyproperty
    def _zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean a cell's
        value is.
        """
        counts = self._counts

        # --- If the matrix is "defective", in the sense that it doesn't have at least
        # --- two rows and two columns that are "full" of data, don't calculate zscores.
        if not np.all(counts.shape) or np.linalg.matrix_rank(counts) < 2:
            return np.full(counts.shape, np.nan)

        residuals = counts - expected_freq(counts)

        # --- variance ---
        rows_margin = self._rows_margin
        columns_margin = self._columns_margin
        table_margin = self.table_margin
        variance = (
            np.outer(rows_margin, columns_margin)
            * np.outer(table_margin - rows_margin, table_margin - columns_margin)
            / table_margin ** 3
        )

        return residuals / np.sqrt(variance)


class _CatXCatMeansMatrix(_CatXCatMatrix):
    """Cat-x-cat matrix for means measure.

    A means matrix has an array of mean values instead of a `counts` array.
    """

    def __init__(self, dimensions, means, unweighted_counts):
        super(_CatXCatMeansMatrix, self).__init__(dimensions, None, unweighted_counts)
        self._means = means

    @lazyproperty
    def columns(self):
        """Sequence of `_MeansVector` object for each valid column element."""
        return tuple(
            _MeansVector(element, unweighted_counts, means)
            for element, unweighted_counts, means in zip(
                self.columns_dimension.valid_elements,
                # --- transpose (.T) arrays to make columns first dimension so columns
                # --- are iterated over by zip.
                self._unweighted_counts.T,
                self._means.T,
            )
        )

    @lazyproperty
    def rows(self):
        """Sequence of `_MeansVector` object for each valid row element."""
        return tuple(
            _MeansVector(element, unweighted_counts, means)
            for element, unweighted_counts, means in zip(
                self.rows_dimension.valid_elements, self._unweighted_counts, self._means
            )
        )


class _MrXCatMatrix(_CatXCatMatrix):
    """Represents MR_X_CAT slices.

    Its `._counts` is a 3D ndarray with axes (rows, selected/not, cols), like:

        [[[ 39  44  24  35]
          [389 447 266 394]]

         [[ 34  36  29  24]
          [394 455 261 405]]

         [[357 415 241 371]
          [ 71  76  49  58]]

         [[  0   0   0   0]
          [428 491 290 429]]]

    Each value is np.float64, or np.int64 if the cube-result is unweighted (as in this
    example).

    Its rows are `_MrOpposingCatVector` objects. Its columns are `_OpposingMrVector`
    objects.
    """

    @lazyproperty
    def columns(self):
        """Sequence of _OpposingMrVector for each columns-dimension element."""
        # --- each `counts` is a 2D ndarray with shape (2, nrows), a selected and
        # --- not-selected array, each of size nrows.
        return tuple(
            _OpposingMrVector(
                counts,
                unweighted_counts,
                element,
                self.table_margin,
                zscores,
                table_std_dev,
                table_std_err,
            )
            for (
                counts,
                unweighted_counts,
                element,
                zscores,
                table_std_dev,
                table_std_err,
            ) in self._column_generator
        )

    @lazyproperty
    def rows(self):
        """Sequence of _MrOpposingCatVector for each rows-dimension element."""
        # --- each `counts` is a 2D ndarray with shape (2, ncols), a selected and
        # --- not-selected array, each of size ncols.
        return tuple(
            _MrOpposingCatVector(
                counts,
                unweighted_counts,
                element,
                table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            )
            for (
                counts,
                unweighted_counts,
                element,
                table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        """1D np.int64 ndarray of unweighted N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct base, since not all of
        the multiple responses were necessarily offered to all respondents. The base for
        each row indicates the number of respondents who were offered that option.
        """
        return np.sum(self._unweighted_counts, axis=(1, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each row of matrix.

        Since the rows-dimension is MR, each row has a distinct table margin, since not
        all of the multiple responses were necessarily offered to all respondents. The
        table-margin for each row indicates the weighted number of respondents who were
        offered that option.

        The values are np.int64 when the source cube-result is unweighted.
        """
        return np.sum(self._counts, axis=(1, 2))

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each row in matrix.

        `._baseline` is the denominator of the column-index and represents the
        proportion of the overall row-count present in each row. A cell with
        a column-proportion exactly equal to this basline will have a column-index of
        100.

        The shape of the return value is (nrows, 1). A baseline for a 4 x 3 matrix looks
        something like this:

            [[0.17935204]
             [0.33454989]
             [0.50762388]
             [0.80331259]
             [0.7996507 ]]

        Baseline is a function of the *unconditional row margin*. Unconditional here
        means that both valid and invalid responses (to the columns-var question) are
        included. This ensures that the baseline is not distorted by a large number of
        missing responses to the columns-question.
        """
        # --- unconditional row-margin is a 1D ndarray of size nrows computed by:
        # --- 1. summing across all columns: np.sum(self._counts_with_missings, axis=2)
        # --- 2. taking only selected counts: [:, 0]
        # --- 3. taking only valid rows: [self._valid_row_idxs]
        uncond_row_margin = np.sum(self._counts_with_missings, axis=2)[:, 0][
            self._valid_row_idxs
        ]
        # --- The "total" (uncond_row_table_margin) is a 1D ndarray of size nrows. Each
        # --- sum includes only valid rows (MR_SUBVAR, axis 0), selected and unselected
        # --- but not missing counts ([0:2]) of the MR_CAT axis (axis 1), and all column
        # --- counts, both valid and missing (axis 2). The rows axis (0) is preserved
        # --- because each MR subvar has a distinct table margin.
        uncond_row_table_margin = np.sum(
            self._counts_with_missings[self._valid_row_idxs][:, 0:2], axis=(1, 2)
        )
        # --- inflate shape to (nrows, 1) for later calculation convenience ---
        return (uncond_row_margin / uncond_row_table_margin)[:, None]

    @property
    def _row_generator(self):
        """Iterable of arguments to row-vector constructor call for each row element.

        Used by `.rows` property. Cannot be a lazyproperty because an iterator is
        exhausted on each use and must be created newly on each call.
        """
        # --- Note `zip()` returns an iterator in Python 3 ---
        return zip(
            self._counts,
            self._unweighted_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._table_std_dev,
            self._table_std_err,
            self._column_index,
        )

    @lazyproperty
    def _table_proportion_variance(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._counts[:, 0, :] / self.table_margin[:, None]
        return p * (1 - p)

    @lazyproperty
    def _table_std_err(self):
        """2D np.float64 ndarray of table-percent std-error for each matrix cell.

        Standard error is sqrt(variance/N).
        """
        return np.sqrt(self._table_proportion_variance / self.table_margin[:, None])

    @lazyproperty
    def _zscores(self):
        """2D np.float64 ndarray of z-score for each matrix cell."""
        return self._array_type_std_res(
            self._counts[:, 0, :],
            self.table_margin[:, None],
            np.sum(self._counts, axis=1),
            np.sum(self._counts[:, 0, :], axis=1)[:, None],
        )


class _MrXCatMeansMatrix(_MrXCatMatrix):
    """MR_X_CAT slice with means measure instead of counts.

    Note that its (weighted) counts are all set to zero. A means slice still has
    meaningful unweighted counts.
    """

    def __init__(self, dimensions, means, unweighted_counts):
        counts = np.zeros(means.shape)
        super(_MrXCatMeansMatrix, self).__init__(dimensions, counts, unweighted_counts)
        self._means = means

    @lazyproperty
    def rows(self):
        """Sequence of _MeansWithMrVector object for each row of matrix."""
        return tuple(
            _MeansWithMrVector(element, unweighted_counts, means[0])
            for element, unweighted_counts, means in zip(
                self.rows_dimension.valid_elements, self._unweighted_counts, self._means
            )
        )


class _CatXMrMatrix(_CatXCatMatrix):
    """Handles CAT x MR slices.

    Its `._counts` is a 3D ndarray with axes (rows, cols, selected/not), like:

        [[[1002.52343241 1247.791605  ]
          [ 765.95079804 1484.36423937]
          [ 656.43937497 1593.87566244]]

         [[1520.23482091 2573.22762247]
          [1291.0925792  2802.36986418]
          [1595.44412365 2498.01831973]]

         [[ 908.65667501 2254.62623471]
          [ 841.76439186 2321.51851785]
          [1603.79596755 1559.48694217]]

         [[ 746.89008236 1753.26322241]
          [ 721.38248086 1778.7708239 ]
          [1255.87038944 1244.28291533]]

         [[   9.83166357   25.9551254 ]
          [   8.23140253   27.55538645]
          [  22.214956     13.57183298]]]

    Each value is np.float64 if the cube-result is weighted (as in this example), or
    np.int64 if unweighted.

    Its rows are `_OpposingMrVector` objects. Its columns are `_MrOpposingCatVector`
    objects.
    """

    def __init__(
        self, dimensions, counts, unweighted_counts, counts_with_missings, overlaps=None
    ):
        super(_CatXMrMatrix, self).__init__(
            dimensions, counts, unweighted_counts, counts_with_missings
        )
        self._overlaps = overlaps

    @lazyproperty
    def columns(self):
        """Sequence of _MrOpposingCatVector object for each column of matrix."""
        return tuple(
            _MrOpposingCatVector(counts.T, unweighted_counts.T, element, table_margin)
            for (
                counts,
                unweighted_counts,
                element,
                table_margin,
            ) in self._column_generator
        )

    @lazyproperty
    def is_augmented(self):
        """True if this matrix comes from an augmented (MR_X_ITSELF aka MR_AUG) cube."""
        return self._overlaps is not None

    @lazyproperty
    def overlaps_tstats(self):
        """Pair of 3D np.float64 ndarray or None when cube is not augmented.

        The array shape is (nrows, ncols, ncols) (ask Ernesto, this is probably changing
        anyway and he will be sure to carefully document the next version :).
        """
        return self._overlaps if self.is_augmented else None

    @lazyproperty
    def rows(self):
        """Sequence of _OpposingMrVector object for each row of matrix."""
        return tuple(
            _OpposingMrVector(
                counts.T,
                unweighted_counts.T,
                element,
                self.table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            )
            for (
                counts,
                unweighted_counts,
                element,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        """1D np.int64 unweighted N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table base.
        """
        # --- unweighted-counts is (nrows, ncols, selected/not) so axis 1 is preserved
        # --- to provide a distinct value for each MR subvar.
        return np.sum(self._unweighted_counts, axis=(0, 2))

    @lazyproperty
    def table_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each column of matrix.

        Because the matrix is X_MR, each column has a distinct table margin.
        """
        # --- unweighted-counts is (nrows, ncols, selected/not) so axis 1 is preserved
        # --- to provide a distinct value for each MR subvar.
        return np.sum(self._counts, axis=(0, 2))

    @lazyproperty
    def _baseline(self):
        """2D np.float64 (or NaN) ndarray of baseline value for each matrix cell.

        Its shape is (nrows, ncols) which corresponds to CAT_X_MR_SUBVAR.

        The baseline value is compared with the column-proportion value for each cell to
        form the column-index value. The baseline is a function of the unconditional row
        margin, which is the sum of counts across both valid and missing columns.

        For CAT_X_MR, `uncond_row_margin` sums across the MR_CAT (selected, not,
        missing) dimension to include missing values (an MR_SUBVAR element is never
        "missing": true).
        """
        # --- counts_with_missings.shape is (nall_rows, ncols, selected/not/missing).
        # --- axes[1] corresponds to the MR_SUBVAR dimension, in which there are never
        # --- "missing" subvars (so nall_cols always equals ncols for that dimension
        # --- type). uncond_row_margin selects only valid rows, retains all columns and
        # --- reduces the selected/not/missing axis by summing those three counts. Its
        # --- shape is (nrows, ncols).
        uncond_row_margin = np.sum(self._counts_with_missings, axis=2)[
            self._valid_row_idxs
        ]
        # --- uncond_table_margin sums across rows, producing 1D array of size ncols,
        # --- (although all its values are always the same).
        uncond_table_margin = np.sum(uncond_row_margin, axis=0)
        # --- division produces a 2D matrix of shape (nrows, ncols) ---
        return uncond_row_margin / uncond_table_margin

    @property
    def _column_generator(self):
        """Iterable providing construction parameters for each column vector in turn.

        Used by `.columns` property. Cannot be a lazyproperty because an iterator is
        exhausted on each use.
        """
        # --- note zip() returns an iterator in Python 3 ---
        return zip(
            np.array([self._counts.T[0].T, self._counts.T[1].T]).T,
            np.array(
                [self._unweighted_counts.T[0].T, self._unweighted_counts.T[1].T]
            ).T,
            self._column_elements,
            self.table_margin,
        )

    @lazyproperty
    def _table_proportion_variance(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._counts[:, :, 0] / self.table_margin
        return p * (1 - p)

    @lazyproperty
    def _zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean each cell's
        value is.
        """
        # if the cube is a special one (5D with MRxItself as last dims)
        # the zscores should be the same as a 2D MRxMR matrix
        return self._array_type_std_res(
            self._counts[:, :, 0],
            self.table_margin,
            np.sum(self._counts[:, :, 0], axis=0),
            np.sum(self._counts, axis=2),
        )


class _CatXMrMeansMatrix(_CatXMrMatrix):
    """Basis for CAT_X_MR slice having mean measure instead of counts."""

    def __init__(self, dimensions, means, unweighted_counts, overlaps=None):
        counts = np.zeros(means.shape)
        super(_CatXMrMeansMatrix, self).__init__(
            dimensions, counts, unweighted_counts, overlaps
        )
        self._means = means

    @lazyproperty
    def rows(self):
        """Sequence of _MeansWithMrVector object for each row of matrix."""
        # --- `._means` is 3D (CAT (nrows), MR_SUBVAR (ncols), MR_CAT (selected/not)).
        # --- Index to take only the selected values for sending to vector, producing
        # --- a 2D array (nrows, ncols) of selected-mean for each matrix cell.
        means_of_selected = self._means[:, :, 0]
        return tuple(
            _MeansWithMrVector(element, unweighted_counts, means)
            for element, unweighted_counts, means in zip(
                self.rows_dimension.valid_elements,
                self._unweighted_counts,
                means_of_selected,
            )
        )


class _MrXMrMatrix(_CatXCatMatrix):
    """Represents MR x MR slices.

    Its `._counts` is a 4D ndarray with axes (rows, selected-and-not, cols,
    selected-and-not), like:

        [[[[2990.03485848 4417.96127006]
           [2713.94318797 4694.05294056]
           [2847.96860219 4560.02752634]]

          [[1198.10181578 3436.90253993]
           [ 914.47846452 3720.52589119]
           [2285.79620941 2349.2081463 ]]]


         [[[2626.08325048 5180.55485426]
           [2396.04310657 5410.59499817]
           [3503.08635211 4303.55175262]]

          [[1562.05342378 2674.30895573]
           [1232.37854592 3003.98383359]
           [1630.67845949 2605.68392002]]]


         [[[3370.04923406 5278.54391705]
           [3033.71862569 5614.87452542]
           [3312.56140096 5336.03175016]]

          [[ 818.0874402  2576.31989293]
           [ 594.7030268  2799.70430633]
           [1821.20341065 1573.20392249]]]


         [[[1822.67560537 2883.99243344]
           [1616.70492531 3089.96311351]
           [1735.59793395 2971.07010487]]

          [[2365.46106889 4970.87137654]
           [2011.71672718 5324.61571825]
           [3398.16687766 3938.16556777]]]]

    Its rows and columns are both `_OpposingMrVector` objects.
    """

    def __init__(
        self,
        dimensions,
        counts,
        unweighted_counts,
        counts_with_missings,
        overlaps=None,
        overlaps_margin=None,
    ):
        super(_MrXMrMatrix, self).__init__(
            dimensions, counts, unweighted_counts, counts_with_missings
        )
        self._overlaps = overlaps
        self._overlaps_margin = overlaps_margin

    @lazyproperty
    def columns(self):
        """Sequence of _OpposingMrVector object for each column in matrix.

        Each column corresponds to a subvar of the second MR dimension.
        """
        return tuple(
            _OpposingMrVector(counts, unweighted_counts, element, table_margin)
            for (
                counts,
                unweighted_counts,
                element,
                table_margin,
            ) in self._column_generator
        )

    @lazyproperty
    def is_augmented(self):
        return True if self._overlaps is not None else False

    @lazyproperty
    def overlaps_tstats(self):
        return self._overlaps if self.is_augmented else None

    @lazyproperty
    def rows(self):
        """Sequence of _OpposingMrVector object for each row of matrix.

        Each row corresponds to a subvar of the first MR dimension.
        """
        return tuple(
            _OpposingMrVector(
                counts[0].T,
                unweighted_counts[0].T,
                element,
                table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            )
            for (
                counts,
                unweighted_counts,
                element,
                table_margin,
                zscores,
                table_std_dev,
                table_std_err,
                column_index,
            ) in self._row_generator
        )

    @lazyproperty
    def table_base(self):
        """2D np.int64 ndarray of distinct unweighted N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-base.
        """
        # --- unweighted_counts is 4D of shape (nrows, 2, ncols, 2):
        # --- (MR_SUBVAR (nrows), MR_CAT (sel/not), MR_SUBVAR (ncols), MR_CAT (sel/not))
        # --- Reduce the second and fourth axes with sum() producing 2D (nrows, ncols).
        # --- This sums (selected, selected), (selected, not), (not, selected) and
        # --- (not, not) cells of the subtable for each matrix cell.
        return np.sum(self._unweighted_counts, axis=(1, 3))

    @lazyproperty
    def table_margin(self):
        """2D np.float/int64 ndarray of weighted N for each cell of matrix.

        Because the matrix is MR_X_MR, each cell corresponds to a 2x2 sub-table
        (selected/not on each axis), each of which has its own distinct table-margin.
        """
        # --- Reduce second and fourth axes (the two MR_CAT dimensions) with sum()
        # --- producing 2D (nrows, ncols). This sums (selected, selected),
        # --- (selected, not), (not, selected) and (not, not) cells of the subtable for
        # --- each matrix cell.
        return np.sum(self._counts, axis=(1, 3))

    @lazyproperty
    def tstats_overlap(self):
        """
        ndarray of correct tstats values considering the overlapped observations
        t = (pi-pj)/sqrt(s.e.ˆ2(pi-pj))
        where
        s.e.ˆ2(pi-pj) = p_i*(1-p_i)/n_i+p_j*(1-p_j)/n_j-2*n_ij*(p_ij-p_i*p_j)/(n_i*n_j)
        ni = base size for first subvar
        nj = base size for second subvar
        nij = number of overlapping observations
        pij = proportion for which both subvar are True (selected)
        In this case MRxMR the diff pi-pj is the pairwise subtraction of the diagonal
        of the shadow_proportions the denominator is the matrix containing the
        unweighted counts of the cube
        """

        # Subtraction of the proportions foreach observation
        diff = (
            np.subtract.outer(
                self._mr_shadow_proportions.diagonal(),
                self._mr_shadow_proportions.diagonal(),
            )
            * -1
        )

        se_pi_pj = np.add.outer(
            self._mr_shadow_proportions.diagonal()
            * (1 - self._mr_shadow_proportions.diagonal())
            / self._overlaps_margin.diagonal(),
            self._mr_shadow_proportions.diagonal()
            * (1 - self._mr_shadow_proportions.diagonal())
            / self._overlaps_margin.diagonal(),
        )

        # Correction factor considering the overlap
        correction_factor = (
            2
            * self._overlaps_margin
            * (
                self._mr_shadow_proportions
                - np.multiply.outer(
                    self._mr_shadow_proportions.diagonal(),
                    self._mr_shadow_proportions.diagonal(),
                )
            )
        ) / np.multiply.outer(
            self._overlaps_margin.diagonal(), self._overlaps_margin.diagonal()
        )
        se_diff = np.sqrt(se_pi_pj - correction_factor)
        np.fill_diagonal(diff, 0)
        np.fill_diagonal(se_diff, 0)
        return diff, se_diff

    @lazyproperty
    def _baseline(self):
        """2D np.float64 ndarray of baseline value for each matrix cell.

        The shape is (nrows, ncols) and all values in a given row are the same. So
        really there are only nrows distinct baseline values, but the returned shape
        makes calculating zscores in a general way more convenient.
        """
        # --- `counts_with_missings` for MR_X_MR is 4D of size (nrows, 3, ncols, 3)
        # --- (MR_SUBVAR, MR_CAT, MR_SUBVAR, MR_CAT). Unconditional row margin:
        # --- * Takes all rows and all cols (axes 0 & 2), because MR_SUBVAR dimension
        # ---   can contain only valid elements (no such thing as "missing": true
        # ---   subvar).
        # --- * Sums selected + unselected + missing categories in second MR_CAT
        # ---   dimension (columns MR, axes[3]). Including missings here fulfills
        # ---   "unconditional" characteristic of margin.
        # --- * Takes only those totals associated with selected categories of first
        # ---   MR_CAT dimension (rows MR). ("counts" for MR are "selected" counts).
        # --- Produces a 2D (nrows, ncols) array.
        uncond_row_margin = np.sum(self._counts_with_missings[:, 0:2], axis=3)[:, 0]
        # --- Unconditional table margin is also 2D (nrows, ncols) but the values for
        # --- all columns in a row have the same value; basically each row has
        # --- a distinct table margin.
        uncond_table_margin = np.sum(self._counts_with_missings[:, 0:2], axis=(1, 3))
        # --- baseline is produced by dividing uncond_row_margin by uncond_table_margin.
        return uncond_row_margin / uncond_table_margin

    @property
    def _column_generator(self):
        """Iterable providing construction parameters for each column vector in turn.

        Used by `.columns` property. Cannot be a lazyproperty because an iterator is
        exhausted on each use.
        """
        # --- note zip() returns an iterator in Python 3 ---
        return zip(
            self._counts.T[0],
            self._unweighted_counts.T[0],
            self._column_elements,
            self.table_margin.T,
        )

    @lazyproperty
    def _mr_shadow_proportions(self):
        """Cube containing item-wise selections, overlap, and nonoverlap
        with all other items in a multiple response dimension, for each
        element of any prepended dimensions:
        A 1d interface to a 4d hypercube of underlying counts.
        """
        return self._counts[:, 0, :, 0] / self._overlaps_margin

    @property
    def _row_generator(self):
        """Iterable providing construction parameters for each column vector in turn.

        Used by `.rows` property. Cannot be a lazyproperty because an iterator is
        exhausted on each use.
        """
        # --- note zip() returns an iterator in Python 3 ---
        return zip(
            self._counts,
            self._unweighted_counts,
            self._row_elements,
            self.table_margin,
            self._zscores,
            self._table_std_dev,
            self._table_std_err,
            self._column_index,
        )

    @lazyproperty
    def _table_proportion_variance(self):
        """2D ndarray of np.float64 table proportion variance for each matrix cell."""
        p = self._counts[:, 0, :, 0] / self.table_margin
        return p * (1 - p)

    @lazyproperty
    def _zscores(self):
        """2D ndarray of np.float64 std-res value for each cell of matrix.

        A z-score is also known as a *standard score* and is the number of standard
        deviations above (positive) or below (negative) the population mean each cell's
        value is.
        """
        return self._array_type_std_res(
            self._counts[:, 0, :, 0],
            self.table_margin,
            np.sum(self._counts, axis=1)[:, :, 0],
            np.sum(self._counts, axis=3)[:, 0, :],
        )


# ===INSERTED (SUBTOTAL) VECTORS===


class _BaseMatrixInsertedVector(object):
    """Base class for a (subtotal) vector inserted in a matrix.

    There are some differences that arise when there are both inserted rows *and*
    inserted columns, which entails the complication of inserted vector *intersections*.

    The `base_rows` and `base_colums` vector collections are already ordered.
    """

    def __init__(self, subtotal, neg_idx, table_margin, base_rows, base_columns):
        self._subtotal = subtotal
        # --- the *negative* idx of this vector among its peer inserted vectors ---
        self._neg_idx = neg_idx
        self._table_margin = table_margin
        self._base_rows = base_rows
        self._base_columns = base_columns

    @lazyproperty
    def addend_idxs(self):
        """ndarray of int base-element offsets contributing to this subtotal.

        Suitable for directly indexing a numpy array object (such as base values or
        margin) to extract the addend values for this subtotal.
        """
        addend_ids = self._subtotal.addend_ids
        return np.fromiter(
            (
                idx
                for idx, vector in enumerate(self._base_vectors)
                if vector.element_id in addend_ids
            ),
            dtype=int,
        )

    @lazyproperty
    def anchor(self):
        """str or int anchor value of this inserted-vector.

        The value is either "top", "bottom", or an int element-id of the base-vector it
        should appear after.
        """
        return self._subtotal.anchor

    @lazyproperty
    def base(self):
        """np.int64 or 1D np.int64 ndarray of unweighted-N for this vector.

        `base` is the unweighted N for the vector. A subtotal vector opposing an MR
        dimension has a distinct base for each cell and produces a 1D np.int64 ndarray.
        A subtotal vector opposing a CAT dimension produces a scalar np.int64 value.
        """
        return np.sum(np.array([v.base for v in self._addend_vectors]), axis=0)

    @lazyproperty
    def column_index(self):
        """1D ndarray of np.nan for each cell of vector.

        Computing the column-index of an inserted vector is hard. Punt for now.
        """
        return np.array([np.nan] * len(self.counts))

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of count subtotal for each vector cell.

        Note that all counts are zero (0.0) for a MEANS matrix.
        """
        # --- addend_vectors are _OrderedVector objects because ordering happens before
        # --- insertion of subtotals.
        return np.sum(np.array([v.counts for v in self._addend_vectors]), axis=0)

    @lazyproperty
    def fill(self):
        """Unconditionally `None` for an inserted vector.

        A `fill` value is normally a str RGB value like "#da09fc", specifying the color
        to use for a chart category or series representing this element. The value
        reflects the resolved element-fill transform cascade. Since an inserted vector
        cannot (currently) have a fill-transform, the default value of `None`
        (indicating "use default color") is unconditionally returned.
        """
        return None

    @lazyproperty
    def hidden(self):
        """True if vector is hidden, either by a hide-transform or pruning.

        An inserted vector can never be hidden explicitly (for now). They can also
        almost never be pruned, except in the case when all of the opposite vectors are
        also pruned (thus leaving no elements for this inserted vector, but also
        a completely empty table).
        """
        return self.pruned

    @lazyproperty
    def is_inserted(self):
        """True when this vector is an inserted vector.

        Unconditionally True for _BaseMatrixInsertedVector.
        """
        return True

    @lazyproperty
    def label(self):
        """str display-name for this vector, for use as its row or column heading."""
        return self._subtotal.label

    @lazyproperty
    def margin(self):
        """np.float/int64 or 1D np.float/int64 ndarray of margin for each vector cell.

        `margin` is the weighted N for the vector. A subtotal vector opposing an MR
        dimension has a distinct margin for each cell and produces a 1D np.float64
        ndarray (or np.int64 if cube-result is unweighted). A subtotal vector opposing
        a CAT dimension produces a scalar np.float/int64 value.
        """
        return np.sum(np.array([v.margin for v in self._addend_vectors]), axis=0)

    @lazyproperty
    def means(self):
        """1D ndarray of NaN value for each cell of vector.

        Means is currently undefined for inserted vectors. np.nan is some sort of
        reasonable value to indicate this.
        """
        return np.full(self.counts.shape, np.nan)

    @lazyproperty
    def numeric_value(self):
        """Unconditionally np.nan; an inserted vector can have no numeric-value."""
        return np.nan

    @lazyproperty
    def opposing_margin(self):
        """Optional 1D np.float/int64 ndarray of weighted N for each opposing vector.

        This value is used in the zscore calculation for a non-MR matrix and
        expected_counts computations. Its value is `None` for an insertion in a means
        matrix (margin is sum of counts and means cube has no counts) or an insertion in
        a matrix with an MR dimension.
        """
        return self._addend_vectors[0].opposing_margin

    @lazyproperty
    def ordering(self):
        """-> (position, index, self) tuple used for interleaving with base vectors.

        This value allows the interleaving of inserted vectors with base vectors to be
        reduced to a sorting operation.

        The int position value is roughly equivalent to the notion of "anchor". It is
        0 for anchor=="top", `sys.maxsize` for anchor=="bottom", and int(anchor) + 1
        otherwise. The +1 ensures inserted vectors appear *after* the vector they are
        anchored to.

        The `index` value is the *negative* index of this subtotal in its collection
        (i.e. the "distance from the end" of this inserted vector). This ensures that an
        inserted vector will always sort *prior* to a base vector with the same position
        while preserving the payload order of the inserted vector when two or more are
        anchored to the same base vector.
        """
        return self._anchor_n, self._neg_idx, self

    @lazyproperty
    def table_margin(self):
        """Scalar np.float/int64 or 1D np.float/int64 ndarray of table weighted N.

        Value is ndarray for an MR vector or one opposing MR. Otherwise (e.g.
        CAT-opposing-CAT) it is a scalar.
        """
        return self._table_margin

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count subtotal for each vector cell."""
        return np.sum(
            np.array([row.unweighted_counts for row in self._addend_vectors]), axis=0
        )

    @lazyproperty
    def zscores(self):
        """1D np.float64 (or np.nan) ndarray of standard-score for each matrix cell."""
        opposing_margin = self.opposing_margin

        # TODO: remove this if statement - temporary hack until MR zscore implementation
        if opposing_margin is None:
            return tuple([np.nan] * len(self.counts))

        margin = self.margin
        table_margin = self.table_margin
        variance = (
            opposing_margin
            * margin
            * ((table_margin - opposing_margin) * (table_margin - margin))
            / table_margin ** 3
        )
        return self._residuals / np.sqrt(variance)

    @lazyproperty
    def _addend_vectors(self):
        """Sequence of base-vectors contributing to this inserted subtotal."""
        return tuple(self._base_vectors[i] for i in self.addend_idxs)

    @lazyproperty
    def _anchor_n(self):
        """Anchor expressed as an int "offset" relative to base-vector indices.

        `anchor_n` represents the base-vector idx *before which* the subtotal should
        appear (even though subtotals appear *after* the row they are anchored to.

        A subtotal with anchor_n `0` appears at the top, one with an anchor of `3`
        appears *before* the base row at offset 3; `sys.maxsize` is used as anchor_n for
        "bottom" anchored subtotals.

        To make this work, the `anchor_n` for a subtotal is idx+1 of the base row it is
        anchored to (for subtotals anchored to a row, not "top" or "bottom"). Combining
        this +1 characteristic with placing subtotals before rows with idx=anchor_n
        produces the right positioning and also allows top and bottom anchors to work
        while representing the position as a single non-negative int.

        See `.ordering` for more.
        """
        anchor = self.anchor

        if anchor == "top":
            return 0
        if anchor == "bottom":
            return sys.maxsize

        anchor = int(anchor)
        for i, vector in enumerate(self._base_vectors):
            if vector.element_id == anchor:
                return i + 1

        # --- default to bottom if target anchor vector not found ---
        return sys.maxsize  # pragma: no cover

    @lazyproperty
    def _base_vectors(self):
        """The base (non-inserted) vector "peers" for this inserted vector.

        This is `._base_rows` for an inserted row or `._base_columns` for an inserted
        column.
        """
        raise NotImplementedError(
            "`._base_vectors` must be implemented by each subclass"
        )  # pragma: no cover

    @lazyproperty
    def _expected_counts(self):
        """1D np.float64 ndarray of expected value of count for each vector cell."""
        return self.opposing_margin * self.margin / self.table_margin

    @lazyproperty
    def _residuals(self):
        """1D np.float64 ndarray of count/expected difference for each vector cell."""
        return self.counts - self._expected_counts


class _InsertedColumn(_BaseMatrixInsertedVector):
    """Represents an inserted (subtotal) column."""

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        An inserted vector can almost never be pruned, except in the case when all of
        the opposite vectors are also pruned (thus leaving no elements for this
        inserted vector and a completely empty table).
        """
        return self._subtotal.prune and not np.any(
            np.array([row.base for row in self._base_rows])
        )

    @lazyproperty
    def _base_vectors(self):
        """The base vectors of an inserted column are the base-columns.

        The base columns are the non-inserted columns from which the addend columns are
        drawn.
        """
        return self._base_columns


class _InsertedRow(_BaseMatrixInsertedVector):
    """Represents an inserted (subtotal) row."""

    @lazyproperty
    def pruned(self):
        """True if vector is pruned.

        An inserted vector can almost never be pruned, except in the case when all of
        the opposite vectors are also pruned (thus leaving no elements for this
        inserted vector and a completely empty table).
        """
        return self._subtotal.prune and not np.any(
            np.array([column.base for column in self._base_columns])
        )

    @lazyproperty
    def _base_vectors(self):
        """The base vectors of an inserted row are the base-rows.

        The base rows are the non-inserted rows from which the addend rows are drawn.
        """
        return self._base_rows


# ===TRANSFORMATION VECTORS===


class _BaseTransformationVector(object):
    """Base class for most transformation vectors."""

    def __init__(self, base_vector):
        self._base_vector = base_vector

    @lazyproperty
    def element_id(self):
        """int identifier of category or subvariable this vector represents."""
        return self._base_vector.element_id

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._base_vector.fill

    @lazyproperty
    def hidden(self):
        """True if this vector should not appear in the transformed matrix.

        A vector can be hidden by a "hide" transform on its dimension element or because
        the vector is "pruned".
        """
        return self._base_vector.hidden

    @lazyproperty
    def is_inserted(self):
        """True if this vector is an inserted subtotal."""
        return self._base_vector.is_inserted

    @lazyproperty
    def label(self):
        """str display-name used for this vector's row or column heading."""
        return self._base_vector.label

    @lazyproperty
    def margin(self):
        """Scalar np.float/int64 or 1D np.float/ing64 ndarray of weighted N for vector.

        A vector that opposes an MR dimension has an array of weighted N values because
        each MR_SUBVAR element has a distinct weighted N count. A vector opposing a CAT
        dimension produces a scalar value. Values are np.int64 if the cube-result is
        unweighted.
        """
        return self._base_vector.margin

    @lazyproperty
    def means(self):
        """1D np.float64 (or np.nan) ndarray of mean value for each vector cell.

        Intended only for use when the cube-result measure is means. Raises
        `AttributeError` when called on a cube-result with count measure (and no means
        measure).
        """
        return self._base_vector.means

    @lazyproperty
    def numeric_value(self):
        """int, float, or np.nan representing numeric value for this vector's element.

        This mapping of a category to a numeric value is optional, but when present
        allows additional quantitative computations to be applied to categorical data,
        in particular, so-called "scale-means".

        Its value may be int or float if present and is np.nan if not specified by user
        or the vector is an inserte subtotal.
        """
        return self._base_vector.numeric_value

    @lazyproperty
    def opposing_margin(self):
        """1D np.float/int64 ndarray (or None) of weighted N for each opposing vector.

        Value can be `None` when cube-result measure is means or involves an MR
        dimension.
        """
        return self._base_vector.opposing_margin

    @lazyproperty
    def table_base(self):
        """np.int64 unweighted N for overall table.

        Cannot be computed at vector level and must be passed in on vector construction.
        """
        return self._base_vector.table_base

    @lazyproperty
    def table_margin(self):
        """Scalar np.float/int64 or 1D np.float/int64 ndarray of weighted N for table.

        A vector that opposes an MR dimension has an array of weighted N values because
        each MR_SUBVAR element has a distinct table margin. A vector opposing a CAT
        dimension produces a scalar value. Values are np.int64 if the cube-result is
        unweighted.
        """
        return self._base_vector.table_margin


class _AssembledVector(_BaseTransformationVector):
    """Vector with base, as well as inserted, elements (of the opposite dimension)."""

    def __init__(self, base_vector, opposite_inserted_vectors, vector_idx):
        super(_AssembledVector, self).__init__(base_vector)
        self._opposite_inserted_vectors = opposite_inserted_vectors
        self._vector_idx = vector_idx

    @lazyproperty
    def base(self):
        """Scalar np.int64 or 1D np.int64 ndarray of unweighted N for this vector.

        A vector that opposes an MR dimension has an array of unweighted N values
        because each MR_SUBVAR element has a distinct unweighted N. A vector opposing
        a CAT dimension produces a scalar value.
        """
        return self._base_vector.base

    @lazyproperty
    def column_index(self):
        """1D np.float64/np.nan ndarray of column-index for each vector cell.

        A cell corresponding to an inserted subtotal gets np.nan.
        """

        def fsubtot(inserted_vector):
            """-> np.nan as unconditional col-index value for `inserted_vector`.

            Called by ._apply_interleaved() to compute inserted value which it places
            in the right vector position.
            """
            # TODO: Replace with real column index values from insertions vectors. This
            # should be something like:
            #   col_ind = (ins1.prop + ins2.prop) / (ins1.baseline + ins2.baseline)
            # ask @mike to confirm
            return np.nan

        return self._apply_interleaved(self._base_vector.column_index, fsubtot)

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of weighted count for each vector cell."""
        base_vector_counts = self._base_vector.counts

        def fsubtot(inserted_vector):
            """-> np.float/int64 count for `inserted_vector`.

            Passed to and called by ._apply_interleaved() to compute inserted value
            which it places in the right vector position.
            """
            return np.sum(base_vector_counts[inserted_vector.addend_idxs])

        return self._apply_interleaved(base_vector_counts, fsubtot)

    @lazyproperty
    def means(self):
        """1D ndarray of np.float64 or np.nan mean value for each vector cell.

        A cell corresponding to an inserted subtotal gets a mean of np.nan.
        """

        def fsubtot(inserted_vector):
            """-> np.nan as unconditional mean value for `inserted_vector`.

            Passed to and called by ._apply_interleaved() to compute inserted value
            which it places in the right vector position.
            """
            return np.nan

        return self._apply_interleaved(self._base_vector.means, fsubtot)

    @lazyproperty
    def proportions(self):
        """1D np.float64/np.nan ndarray of count proportion for each vector cell.

        A cell value is np.nan if its corresponding margin value is zero.
        """
        return self.counts / self.margin

    @lazyproperty
    def pvals(self):
        """1D np.float64/np.nan ndarray of p-value for each vector cell."""
        return 2 * (1 - norm.cdf(np.abs(self.zscores)))

    @lazyproperty
    def table_std_dev(self):
        """1D np.float64 ndarray of std-dev of table-percent for each vector cell."""
        return np.sqrt(self._table_proportion_variance)

    @lazyproperty
    def table_std_err(self):
        """1D np.float64 ndarray of std-err of table-percent for each vector cell."""
        return np.sqrt(self._table_proportion_variance / self.table_margin)

    @lazyproperty
    def table_proportions(self):
        """1D np.float64 ndarray of table-proportion for each vector cell.

        Also known as "cell-proportion", the proportion of overall weighted N for the
        table that appears in that particular cell.
        """
        return self.counts / self._base_vector.table_margin

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted count for each base element and insertion.

        Subtotal values are interleaved with base values in their specified location.
        """
        unweighted_counts = self._base_vector.unweighted_counts

        def fsubtot(inserted_vector):
            """-> np.int64 count for `inserted_vector`.

            Passed to and called by ._apply_interleaved() to compute inserted value
            which it places in the right vector position.
            """
            return np.sum(unweighted_counts[inserted_vector.addend_idxs])

        return self._apply_interleaved(unweighted_counts, fsubtot)

    @lazyproperty
    def zscores(self):
        """1D np.float64/np.nan ndarray of z-score for each base element and insertion.

        Subtotal values are interleaved with base values in their specified location.
        """

        def fsubtot(inserted_vector):
            """-> np.float64 zscore for `inserted_vector`.

            Passed to and called by ._apply_interleaved() to compute inserted value
            which it places in the right vector position.
            """
            # --- when this vector is not itself a subtotal, `inserted_vector` can
            # --- provide the value directly
            if not self.is_inserted:
                return inserted_vector.zscores[self._vector_idx]

            # --- but when this vector IS itself a subtotal, this cell is an
            # --- *intersection* cell and is requires a more sophisticated computation
            margin, table_margin = self.margin, self.table_margin
            opposite_margin = np.sum(self.opposing_margin[inserted_vector.addend_idxs])
            variance = (
                opposite_margin
                * margin
                * ((table_margin - opposite_margin) * (table_margin - margin))
                / table_margin ** 3
            )
            expected_count = opposite_margin * margin / table_margin
            cell_value = np.sum(self._base_vector.counts[inserted_vector.addend_idxs])
            residuals = cell_value - expected_count
            return residuals / np.sqrt(variance)

        return self._apply_interleaved(self._base_vector.zscores, fsubtot)

    def _apply_interleaved(self, base_values, fsubtot):
        """-> 1D array of result of applying fbase or fsubtot to each interleaved item.

        `base_values` is the "unassembled" vector measure values.

        `fsubtot(inserted_vector)` :: inserted_vector -> intersection_value

        Takes care of the details of getting vector "cells" interleaved in the right
        order, you just provide the "unassembled" values and a function to apply to each
        subtotal-vector to get its value.
        """
        subtotals = self._opposite_inserted_vectors

        return np.array(
            tuple(
                fsubtot(subtotals[idx]) if idx < 0 else base_values[idx]
                for idx in self._interleaved_idxs
            )
        )

    @lazyproperty
    def _interleaved_idxs(self):
        """-> tuple of int: idx for base and inserted values, in display order.

        Inserted value indicies are negative, to distinguish them from base vector
        indices. The indexes are interleaved simply by sorting their orderings. An
        ordering is a (position, idx) pair.
        """

        def iter_insertion_orderings():
            """Generate (int: position, int: idx) for each opposing insertion.

            The position for an insertion is an int representation of its anchor and its
            idx is the *negative* offset of its position in the opposing insertions
            sequence (like -3, -2, -1 for a sequence of length 3). The negative idx
            works just as well as the normal one for accessing the subtotal but insures
            that an insertion at the same position as a base row always sorts *before*
            the base row.

            The `position` int for a subtotal is 0 for anchor "top", sys.maxsize for
            anchor "bottom", and int(anchor) + 1 for all others. The +1 ensures
            a subtotal appears *after* the vector it is anchored to.
            """
            for v in self._opposite_inserted_vectors:
                pos, idx, _ = v.ordering
                yield pos, idx

        def iter_base_value_orderings():
            """Generate (int: position, int: idx) for each base-vector value.

            The position of a base value is simply it's index in the vector.
            """
            for idx in range(len(self._base_vector.counts)):
                yield idx, idx

        return tuple(
            idx
            for pos, idx in sorted(
                itertools.chain(iter_insertion_orderings(), iter_base_value_orderings())
            )
        )

    @lazyproperty
    def _table_proportion_variance(self):
        """1D ndarray of np.float64 table proportion variance for each vector cell."""
        p = self.table_proportions
        return p * (1 - p)


class _VectorAfterHiding(_BaseTransformationVector):
    """Reflects a row or column with hidden elements removed."""

    def __init__(self, base_vector, opposite_vectors):
        super(_VectorAfterHiding, self).__init__(base_vector)
        self._opposite_vectors = opposite_vectors

    @lazyproperty
    def base(self):
        """np.int64 or 1D np.int64 ndarray of unweighted-N for this vector.

        Unweighted N values are a scalar when this vector opposes a CAT dimension but
        are an array when it opposes an MR dimension.
        """
        if not isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base
        return self._base_vector.base[self._visible_element_idxs]

    @lazyproperty
    def column_index(self):
        """1D np.float64/np.nan ndarray of column-index for each visible vector cell."""
        return self._base_vector.column_index[self._visible_element_idxs]

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of weighted-count for each visible vector cell."""
        return self._base_vector.counts[self._visible_element_idxs]

    @lazyproperty
    def margin(self):
        """np.float/int64 or 1D np.float/int64 ndarray of weighted-N for this vector.

        Weighted-N values are a scalar when this vector opposes a CAT dimension but
        are an array when it opposes an MR dimension.
        """
        if not isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin
        return self._base_vector.margin[self._visible_element_idxs]

    @lazyproperty
    def means(self):
        """1D np.float64/np.nan ndarray of mean value for each visible vector cell.

        Cell value is `np.nan` for each cell corresponding to an inserted subtotal
        (mean of addend cells cannot simply be added to get the mean of the subtotal).
        """
        return self._base_vector.means[self._visible_element_idxs]

    @lazyproperty
    def proportions(self):
        """1D np.float64/np.nan ndarray of vector-proportion for each vector cell.

        vector-proportions is column-proportions when the vector is a column and
        row-proportions when it represents a row. The cell values are `np.nan` when the
        vector-margin is zero (producing a division by zero).
        """
        return self._base_vector.proportions[self._visible_element_idxs]

    @lazyproperty
    def pvals(self):
        """1D np.float64/np.nan ndarray of p-value for each visible vector cell.

        A cell value representing an inserted subtotal can be `np.nan` in certain
        situations involving MR dimensions.
        """
        return self._base_vector.pvals[self._visible_element_idxs]

    @lazyproperty
    def table_std_dev(self):
        """1D np.float64 ndarray of std-dev of table-proportion for each vector cell."""
        return self._base_vector.table_std_dev[self._visible_element_idxs]

    @lazyproperty
    def table_std_err(self):
        """1D np.float64 ndarray of std-err of table-proportion for each vector cell."""
        return self._base_vector.table_std_err[self._visible_element_idxs]

    @lazyproperty
    def table_proportions(self):
        """1D np.float64 ndarray of table-proportion for each visible vector cell.

        Table-proportion is the proportion of the weighted-N for the overall table
        represented in the cell. The table-margin (table weighted-N) for an MR dimension
        is distinct for each MR subvar, so each table-proportion is not necessarily
        computed with the same denominator value.
        """
        return self._base_vector.table_proportions[self._visible_element_idxs]

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each visible vector cell."""
        return self._base_vector.unweighted_counts[self._visible_element_idxs]

    @lazyproperty
    def zscores(self):
        """1D np.float64/np.nan ndarray of z-score for each visible vector cell.

        A cell value representing an inserted subtotal can be `np.nan` in certain
        situations involving MR dimensions.
        """
        return self._base_vector.zscores[self._visible_element_idxs]

    @lazyproperty
    def _visible_element_idxs(self):
        """An 1D ndarray of int idxs of non-hidden values, suitable for indexing.

        This value is derived from the opposing vectors collection, based on the hidden
        status of its elements.
        """
        return np.array(
            [
                idx
                for idx, opposite_vector in enumerate(self._opposite_vectors)
                if not opposite_vector.hidden
            ],
            dtype=int,
        )


class _OrderedVector(_BaseTransformationVector):
    """Rearranges its "cells" to the order of its opposing-vectors.

    For example, the cells in a row appear in the order of the column vectors. Ordering
    is performed before insertions, so this vector includes only base-vector values.

    `opposing_order` is a 1D int ndarray of indices of opposing dimension elements, in
    the order those elements (and their vectors) appear in the opposing dimension.
    """

    def __init__(self, base_vector, opposing_order, index):
        super(_OrderedVector, self).__init__(base_vector)
        self._opposing_order = opposing_order
        self._index = index

    @lazyproperty
    def base(self):
        """np.int64 or 1D np.int64 ndarray of unweighted-N for this vector.

        Values appear in the order of their corresponding opposing vector. Unweighted
        N values are a scalar when this vector opposes a CAT dimension but are an array
        when it opposes an MR dimension.
        """
        if isinstance(self._base_vector.base, np.ndarray):
            return self._base_vector.base[self._opposing_order]
        return self._base_vector.base

    @lazyproperty
    def column_index(self):
        """1D np.float64 ndarray of col-index values in opposing-dimension order."""
        return self._base_vector.column_index[self._opposing_order]

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of weighted-counts in opposing-dimension order.

        Values are np.int64 when the cube-result is unweighted.
        """
        return self._base_vector.counts[self._opposing_order]

    @lazyproperty
    def label(self):
        """str display-name for this vector, for use as its row or column heading."""
        return self._base_vector.label

    @lazyproperty
    def margin(self):
        """np.float/int64 or 1D np.float/int64 ndarray of margin for each vector cell.

        Values appear in the order of the opposing dimension. `margin` is the weighted
        N for the vector. A vector opposing an MR dimension has a distinct margin for
        each cell and produces a 1D np.float64 ndarray (or np.int64 if cube-result is
        unweighted). A vector opposing a CAT dimension produces a scalar np.float/int64
        value.
        """
        if isinstance(self._base_vector.margin, np.ndarray):
            return self._base_vector.margin[self._opposing_order]
        return self._base_vector.margin

    @lazyproperty
    def ordering(self):
        """3-tuple (position, index, self) used for interleaving inserted vectors.

        This value allows the interleaving of base and inserted vectors to be reduced to
        a sorting operation.

        The position and index of a base vector are both its index within its ordered
        collection. The `position` value of the ordering is operative for inserted
        vectors.
        """
        return (self._index, self._index, self)

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted-count for each vector cell.

        Values are rearranged (from base-vector position) to appear in the order of the
        opposing dimension.
        """
        return self._base_vector.unweighted_counts[self._opposing_order]

    @lazyproperty
    def zscores(self):
        """1D np.float64/np.nan ndarray of zscore for each vector cell.

        Values are rearranged (from base-vector position) to appear in the order of the
        opposing dimension. A zscore can be `np.nan` in certain situations.
        """
        return self._base_vector.zscores[self._opposing_order]


# ===OPERAND VECTORS===


class _BaseVector(object):
    """Base class for all base-vector objects.

    A "base" vector is one containing all the valid data items for that vector in the
    cube-result, in the order they appear in the cube result. This vector serves as the
    "base" on which other transforms such as reordering, insertions, and hiding are
    performed.

    A vector represents a row or column of data in the overall data matrix. It composes
    the element that corresponds to that row or column and so knows the name,
    element_id, numeric value, etc. of that row or column.
    """

    def __init__(self, element, unweighted_counts):
        self._element = element
        self._unweighted_counts = unweighted_counts

    @lazyproperty
    def base(self):
        """np.int64 or 1D np.int64 ndarray of unweighted-N for this vector.

        A vector opposing an MR dimension has a distinct base for each cell and produces
        a 1D np.int64 ndarray. A vector opposing a CAT dimension has a single base and
        produces a scalar np.int64 value.
        """
        # --- The `axis=0` parameter makes this work for both the X_MR and X_CAT cases.
        # --- A vector opposing an MR dimension (X_MR) has a 2D unweighted_counts like:
        #
        #      [[653  128  120  389  469]
        #       [101   95   96   69   61]]
        #
        # --- where the first axis is selected/unselected and the second dimension is
        # --- a distinct value for each MR-subvar. The sum must include both selected
        # --- and unselected values, but cannot sum across subvars. So the result is
        # --- a 1D array for the X_MR case.
        # ---
        # --- In the X_CAT case, unweighted_counts is a 1D array like:
        #
        #      [246  61  102  318  469]
        #
        # --- Summing across axis-0 gives the right (scalar) result here too, even
        # --- though specifying the axis is superfluous for this case.
        return np.sum(self._unweighted_counts, axis=0)

    @lazyproperty
    def element_id(self):
        """int identifier of category or subvariable this vector represents."""
        return self._element.element_id

    @lazyproperty
    def fill(self):
        """str RGB color like "#def032" or None when not specified.

        The value reflects the resolved element-fill transform cascade. A value of
        `None` indicates no element-fill transform was specified and the default
        (theme-specified) color should be used for this element.
        """
        return self._element.fill

    @lazyproperty
    def hidden(self):
        """True if vector is hidden.

        Vectors are hidden in two ways:

            1. Explicitly via a "hide" transform on its element
            2. Implicitly when its base (unweighted-N) is 0 (also called pruning)

        This property checks whether a vector needs to be hidden, either implicitly or
        explicitly.
        """
        return self._element.is_hidden or (self._element.prune and self.pruned)

    @lazyproperty
    def is_inserted(self):
        """Unconditionally False for a base-vector.

        Only insertion vectors are inserted.
        """
        return False

    @lazyproperty
    def label(self):
        """str display-name for this vector, for use as its row or column heading."""
        return self._element.label

    @lazyproperty
    def numeric_value(self):
        """int, float, or np.nan representing numeric value of this vector's element.

        This mapping of a category to a numeric value is optional, but when present
        allows additional quantitative computations to be applied to categorical data,
        in particular, so-called "scale-means".

        Its value may be int or float if present and is np.nan if not specified by user.
        """
        return self._element.numeric_value

    @lazyproperty
    def pruned(self):
        """True if this vector contains no samples."""
        return self.base == 0


class _CategoricalVector(_BaseVector):
    """Vector for CAT vector that does not oppose an MR dimension.

    Also serves as the base-class for several other vector types that share many of its
    properties.
    """

    def __init__(
        self,
        counts,
        unweighted_counts,
        element,
        table_margin,
        zscores=None,
        table_std_dev=None,
        table_std_err=None,
        column_index=None,
        opposing_margin=None,
    ):
        super(_CategoricalVector, self).__init__(element, unweighted_counts)
        self._counts = counts
        self._table_margin = table_margin
        self._zscores = zscores
        self._table_std_dev = table_std_dev
        self._table_std_err = table_std_err
        self._column_index = column_index
        self._opposing_margin = opposing_margin

    @lazyproperty
    def column_index(self):
        """1D np.float64/np.nan ndarray of column-index for each vector cell.

        Column-index must be computed at the matrix level, which passes those values to
        the vector on construction.
        """
        return self._column_index

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of weighted count for each vector cell."""
        return self._counts

    @lazyproperty
    def margin(self):
        """Scalar np.float/int64 of unweighted-N for this vector.

        This property must be overridden for vectors that oppose an MR dimension and
        produce a distinct margin value for each vector cell.
        """
        return np.sum(self._counts)

    @lazyproperty
    def opposing_margin(self):
        """1D np.float/int64 ndarray of weighted-N for each opposing dimension vector.

        This value is computed at the matrix level and passed in on construction. It can
        be `None` in certain circumstances.
        """
        return self._opposing_margin

    @lazyproperty
    def proportions(self):
        """1D np.float64/np.nan ndarray of (weighted) count proportions for vector.

        Each value is between 0.0 and 1.0 and indicates the fraction of overall vector
        counts that appear in that cell. A cell value is np.nan if its corresponding
        margin value is zero. Note that when this vector opposes an MR dimension the
        vector margin is distinct for each cell and the proportions may not sum to 1.0.
        """
        return self.counts / self.margin

    @lazyproperty
    def table_margin(self):
        """Scalar np.float/int64 or 1D np.float/int64 ndarray of table weighted N.

        Table margin must be computed at the matrix level, which passes this value to
        the vector on construction. When this vector opposes an MR dimension, it's table
        dimension is an array (each vector cell has a distinct table-margin value).
        A scalar value is returned when the opposing dimension is CAT.
        """
        return self._table_margin

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted count for each vector cell."""
        return self._unweighted_counts

    @lazyproperty
    def zscores(self):
        """1D np.float64/np.nan ndarray of z-score for each vector cell."""
        return self._zscores


class _MeansVector(_BaseVector):
    """Used on a CAT dimension of a CAT_X_CAT matrix with means cube-result measure."""

    def __init__(self, element, unweighted_counts, means):
        super(_MeansVector, self).__init__(element, unweighted_counts)
        self._unweighted_counts = unweighted_counts
        self._means = means

    @lazyproperty
    def counts(self):
        """1D ndarray of np.nan for each vector cell.

        A cube-result with means measure has no weighted counts (or they are ignored if
        they happen to be present, which is very unusual).
        """
        return np.full(self._means.shape, np.nan)

    @lazyproperty
    def means(self):
        """1D np.float64/np.nan ndarray of mean for each vector cell.

        Mean is np.nan for a cell with an unweighted-count of zero.
        """
        return self._means

    @lazyproperty
    def margin(self):
        """Unconditionally np.nan for a means vector.

        A means vector has no (weighted) counts, so it can have no margin.
        """
        return np.nan


class _MeansWithMrVector(_MeansVector):
    """Used for row vectors in a means matrix with an MR dimension."""

    # TODO: Work out why non-means vectors are used for the columns of these matrices.
    # I expect it is because _Slice gets most of its values from rows and columns are
    # only used for column-oriented things like column-proportions and column-labels.
    # Still, could be bugs lurking here as to how columns handle means.

    @lazyproperty
    def base(self):
        """np.int64 unweighted-N for this vector."""
        return np.sum(self._unweighted_counts[0])


class _MrOpposingCatVector(_CategoricalVector):
    """Used for multiple-response dimension when opposing dimension is categorical.

    It is constructed from a 2D np.float/int64 `counts` array with axes
    (selected/not, cells) like:

        [[  6.53222713  12.85476298  12.0520568   38.91870264  46.988204  ]
         [101.92961604  95.60708018  96.40978636  69.54314052  61.47363916]]

    This vector is suitable for use by either a CAT-opposing-MR vector or an
    MR-opposing-MR vector since both have the same number of values and the axis layout
    can be made uniform by the caller.
    """

    def __init__(
        self,
        counts,
        unweighted_counts,
        label,
        table_margin,
        zscores=None,
        table_std_dev=None,
        table_std_err=None,
        column_index=None,
    ):
        super(_MrOpposingCatVector, self).__init__(
            counts[0],
            unweighted_counts[0],
            label,
            table_margin,
            zscores,
            table_std_dev,
            table_std_err,
            column_index,
        )
        self._both_selected_and_unselected_counts = counts
        self._both_selected_and_unselected_unweighted_counts = unweighted_counts

    @lazyproperty
    def pruned(self):
        """True if this vector contains no samples."""
        vector_base = np.sum(self._both_selected_and_unselected_unweighted_counts)
        return vector_base == 0

    @lazyproperty
    def table_base(self):
        """np.int64 "table" unweighted-N for this vector.

        Because this MR vector opposes a CAT dimension, all its cells share a single
        table-base value. Note however that its sibling MR vectors each have their own
        distinct table-base which is often different. This is because each MR response
        may be presented to a different number of respondents.
        """
        return np.sum(self._both_selected_and_unselected_unweighted_counts)

    @lazyproperty
    def table_margin(self):
        """np.float/int64 "table" weighted-N for this vector.

        Because this MR vector opposes a CAT dimension, a single table-margin value is
        shared by all cells in this vector. Note however that its sibling MR vectors
        each have their own distinct table-margin which is often different.
        """
        return np.sum(self._both_selected_and_unselected_counts)


class _OpposingMrVector(_CategoricalVector):
    """CAT or MR vector that opposes an MR dimension.

    It is constructed from `counts` that are a 2D np.float/int64 vector with axes
    (selected/not, cells) like:

        [[  6.53222713  12.85476298  12.0520568   38.91870264  46.988204  ]
         [101.92961604  95.60708018  96.40978636  69.54314052  61.47363916]]

    This vector is suitable for use by either a CAT-opposing-MR vector or an
    MR-opposing-MR vector since both have the same number of values and the axis layout
    can be made uniform by the caller.
    """

    @lazyproperty
    def counts(self):
        """1D np.float/int64 ndarray of weighted count for each vector cell."""
        return self._counts[0, :]

    @lazyproperty
    def margin(self):
        """1D np.float/int64 ndarray of weighted-N for each vector cell.

        Because the opposing dimension is MR, each vector cell has a distinct margin.
        """
        return np.sum(self._counts, axis=0)

    @lazyproperty
    def pruned(self):
        """True if this vector contains no samples."""
        vector_base = np.sum(self._unweighted_counts, axis=0)
        return (vector_base == 0).all()

    @lazyproperty
    def unweighted_counts(self):
        """1D np.int64 ndarray of unweighted selected-count for each vector cell."""
        return self._unweighted_counts[0, :]
