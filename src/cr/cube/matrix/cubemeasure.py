# encoding: utf-8

"""Provides abstracted cube-measure objects used as the basis for second-order measures.

There are several cube-measures that can appear in a cube-response, including
unweighted-counts, weighted-counts (aka. counts), means, and others.
"""

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class CubeMeasures:
    """Provides access to all cube-measure objects for this cube-result."""

    def __init__(self, cube, dimensions, slice_idx):
        self._cube = cube
        self._dimensions = dimensions
        self._slice_idx = slice_idx

    @lazyproperty
    def cube_means(self):
        """_BaseCubeMeans subclass object for this cube-result."""
        return _BaseCubeMeans.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_overlaps(self):
        """_BaseCubeOverlaps subclass object for this cube-result."""
        return _BaseCubeOverlaps.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_sum(self):
        """_BaseCubeSums subclass object for this cube-result."""
        return _BaseCubeSums.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def cube_stddev(self):
        """_BaseCubeStdDev subclass object for this cube-result."""
        return _BaseCubeStdDev.factory(self._cube, self._dimensions, self._slice_idx)

    @lazyproperty
    def unconditional_cube_counts(self):
        """_BaseUnconditionalCounts subclass object for this cube-result."""
        return _BaseUnconditionalCubeCounts.factory(
            self._cube, self._dimensions, self._slice_idx
        )

    @lazyproperty
    def unweighted_cube_counts(self):
        """_BaseWeightedCounts subclass object for this cube-result."""
        valid_counts = self._cube.unweighted_valid_counts
        counts = (
            valid_counts if valid_counts is not None else self._cube.unweighted_counts
        )
        diff_nans = True if valid_counts is not None else False

        return _BaseCubeCounts.factory(
            counts, diff_nans, self._cube, self._dimensions, self._slice_idx
        )

    @lazyproperty
    def weighted_cube_counts(self):
        """_BaseWeightedCounts subclass object for this cube-result."""
        valid_counts = self._cube.weighted_valid_counts
        counts = valid_counts if valid_counts is not None else self._cube.counts
        diff_nans = True if valid_counts is not None else False

        return _BaseCubeCounts.factory(
            counts, diff_nans, self._cube, self._dimensions, self._slice_idx
        )


class _BaseCubeMeasure:
    """Base class for all cube-measure objects."""

    def __init__(self, dimensions):
        self._dimensions = dimensions

    @classmethod
    def _slice_idx_expr(cls, cube, slice_idx):
        """Return np.s_ advanced-indexing slice object to extract values for slice_idx.

        `cube` can contain data for multiple `_Slice` objects. The returned `numpy`
        advanced-indexing expression selects only those values of `cube` that pertain
        to the slice indicated by `slice_idx`.
        """
        # --- for a 2D cube we take the whole thing (1D is not expected here) ---
        if cube.ndim < 3:
            return np.s_[:]

        # --- if 0th dimension of a >2D cube is MR, we only take the "Selected" portion
        # --- of the indicated initial-MR subvar, because the slice is to represent the
        # --- values for "respondents who selected" that MR response (and not those who
        # --- didn't select it or did not respond).
        if cube.dimension_types[0] == DT.MR:
            return np.s_[slice_idx, 0]

        # --- for other 3D cubes we just select the 2D "table" portion associated with
        # --- the `slice_idx`-th table dimension element.
        return np.s_[slice_idx]


# === COUNTS (UNWEIGHTED & WEIGHTED) ===


class _BaseCubeCounts(_BaseCubeMeasure):
    """Base class for count (weighted & unweighted) cube-measure variants."""

    def __init__(self, dimensions, counts, diff_nans):
        super(_BaseCubeCounts, self).__init__(dimensions)
        self._counts = counts
        self._diff_nans = diff_nans

    @classmethod
    def factory(cls, counts, diff_nans, cube, dimensions, slice_idx):
        """Return _BaseCubeCounts subclass instance appropriate to `cube`

        Chooses between unweighted and weighted counts based on `type`.
        """
        dimension_type_strings = tuple(
            "MR"
            if dim_type == DT.MR
            else "ARR"
            if dim_type in DT.ARRAY_TYPES
            else "CAT"
            for dim_type in cube.dimension_types[-2:]
        )
        CubeCountsCls = {
            ("MR", "MR"): _MrXMrCubeCounts,
            ("MR", "ARR"): _MrXArrCubeCounts,
            ("MR", "CAT"): _MrXCatCubeCounts,
            ("ARR", "MR"): _ArrXMrCubeCounts,
            ("ARR", "ARR"): _ArrXArrCubeCounts,
            ("ARR", "CAT"): _ArrXCatCubeCounts,
            ("CAT", "MR"): _CatXMrCubeCounts,
            ("CAT", "ARR"): _CatXArrCubeCounts,
        }.get(dimension_type_strings, _CatXCatCubeCounts)

        return CubeCountsCls(
            dimensions, counts[cls._slice_idx_expr(cube, slice_idx)], diff_nans
        )

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.column_bases`"
        )

    @lazyproperty
    def columns_base(self):
        """Optional 1D np.float64 ndarray of column-wise base for each column.

        This marginal is used as the (un)weighted N row that is shown (typically)
        below other measures.

        This value is None if the columns dimension is not CAT because in that case each
        row has a unique rows base, and so it is not possible to reduce to a 1D
        vector in the columns orientation (aka a single row). Subclasses with legitimite
        values (eg *_X_CAT) must override this default method.
        """
        return None

    @lazyproperty
    def columns_pruning_mask(self):
        """1D bool np.ndarray indicating whether all cells in column are empty

        The columns-pruning-mask indicates when every cell in a column has base values
        of 0. When the column is *not* an MR, the base used is equal to the columns_base,
        but when the column is an MR, we use the sum of selected & non-selected values from
        the column, whereas the columns_base uses just the selected values.
        """
        # --- Bases are positive, so we can sum them to see if all of them are equal to 0
        return self._columns_pruning_base == 0

    @lazyproperty
    def columns_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each column.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, but each component is meaningful:

        * "columns": Indicates it is a marginal in the "columns" orientation (kind of
          like a stripe in the shape of a row).
        * "table": Indicates that it is the base for the whole table. When the
          `.table_base` exists (CAT X CAT), it is a repetition of that, but when
          the rows are array (and therefore we can't sum across them), each cell has
          its own value.
        * "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)

        This value is None if the columns dimension is not CAT because in that case each
        row has a unique table base, and so it is not possible to reduce to a 1D
        vector in the columns orientation (aka a single row). Subclasses with legitimite
        values (eg *_X_CAT) must override this default method.
        """
        return None

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.counts`"
        )

    @lazyproperty
    def diff_nans(self):
        """Boolean, indicates if subtotal differences (rows and cols) has to be NaN.

        We set the subtotal differences to be NaN for count measures that are paired
        alongside numeric dimension collapsing measures (eg mean/count/etc.) because
        those measures are undefined for subtotal differences. We want to do this here
        because we want to make sure that all calculations derived from the counts are
        also correctly set to NaN, rather than having to track down all the derived
        calculations.

        The parameter must be passed in during _BaseCubeCount's creation because it has
        no knowledge of what type of counts it's using, and the way to tell if the counts
        are paired with a numeric variable is by knowing if it comes from `*_counts`
        or `*_valid_counts`.
        """
        return self._diff_nans

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.row_bases`"
        )

    @lazyproperty
    def rows_base(self):
        """Optional 1D np.float64 ndarray of row-wise base for each row.

        This marginal is used as the (un)weighted N column that is shown alongside other
        measures.

        This value is None if the rows dimension is not CAT because in that case each
        column has a unique rows base, and so it is not possible to reduce to a 1D
        vector in the rows orientation (aka a single column). Subclasses with legitimite
        values (eg CAT_X_*) must override this default method.
        """
        return None

    @lazyproperty
    def rows_pruning_mask(self):
        """1D bool np.ndarray indicating whether all cells in rows are empty

        The rows-pruning-mask indicates when every cell in a row has base values
        of 0. When the row is *not* an MR, the base used is equal to the rows_base,
        but when the row is an MR, we use the sum of selected & non-selected values from
        the row, whereas the rows_base uses just the selected values.
        """
        # --- Bases are positive, so we can sum them to see if all of them are equal
        # --- to 0.
        return self._rows_pruning_base == 0

    @lazyproperty
    def rows_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each row.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, but each component is meaningful:

        * "rows": Indicates it is a marginal in the "rows" orientation (kind of
          like a stripe in the shape of a row).
        * "table": Indicates that it is the base for the whole table. When the
          `.table_base` exists (CAT X CAT), it is a repetition of that, but when
          the columns are array (and therefore we can't sum across them), each cell has
          its own value.
        * "base": Indicates that it is the base, not necessarily the counts (eg the sum
          of selected and non-selected for MR variables)

        This value is None if the rows dimension is not CAT because in that case each
        column has a unique table base, and so it is not possible to reduce to a 1D
        vector in the rows orientation (aka a single column). Subclasses with legitimite
        values (eg CAT_X_*) must override this default method.
        """
        return None

    @lazyproperty
    def table_base(self):
        """Optional float of the table-wise base for the whole table

        None if either dimension is an ARRAY or MR because the table base varies.
        The subclass with a legitimite value (CAT_X_CAT) must override this default
        method.
        """
        return None

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.table_bases`"
        )

    @lazyproperty
    def _columns_pruning_base(self):
        """1D bool np.ndarray of the sum of the column-bases for cells in a column

        Used to calculate the columns-pruning-mask, which indicates when every cell in a
        column has base values of 0. When the column is *not* an MR, the base used is
        equal to the columns_base, but when the column is an MR, we use the sum of
        selected & non-selected values the column, whereas the columns_base uses just the
        selected values. These values are not meaningful on their own, but used to
        calculate the columns-pruning-mask.

        This method works when the column is not an MR, and must be overriden when it is
        MR.
        """
        # --- Bases are positive, so we can sum them to see if all of them are equal to 0
        # --- To really hammer home the point that this number is invalid, we keep it
        # --- private. But it's easier to test, so it is separate from the mask.
        return np.sum(self.column_bases, axis=0)

    @lazyproperty
    def _rows_pruning_base(self):
        """1D bool np.ndarray of the sum of the row-bases for cells in a row

        Used to calculate he rows-pruning-mask, which indicates when every cell in a row
        of 0. When the row is *not* an MR, the base used is equal to the rows_base,
        but when the row is an MR, we use the sum of selected & non-selected values from
        the row, whereas the rows_base uses just the selected values. These values
        are not meaningful on their own, but used to calculate the rows-pruning-mask.

        This method works when the row is not an MR, and must be overriden when it is MR.
        """
        # --- Bases are positive, so we can sum them to see if all of them are equal to 0
        # --- To really hammer home the point that this number is invalid, we keep it
        # --- private. But it's easier to test, so it is separate from the mask.
        return np.sum(self.row_bases, axis=1)


class _ArrXArrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=ARR & columns=ARR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, bases are equal to counts
        return self.counts

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- No MR, so counts are already in correct shape
        return self._counts

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, bases are equal to counts
        return self.counts

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, bases are equal to counts
        return self.counts


class _ArrXCatCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=ARR & columns=CAT dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, bases are equal to counts
        return self.counts

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- No MR, so counts are already in correct shape
        return self._counts

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        return np.broadcast_to(self.rows_base[:, None], self.counts.shape)

    @lazyproperty
    def rows_base(self):
        """Optional 1D np.float64 ndarray of row-wise base for each row.

        This marginal is used as the (un)weighted N column that is shown alongside other
        measures.
        """
        # --- Avaialable because column is CAT, equal to the sum across cols dimension.
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def rows_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each row.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.rows_table_base` for details.
        """
        # --- Available because columns are CAT, equal to the rows_base because the row
        # --- is array and so addition over rows is not possible
        return self.rows_base

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- Can't sum across rows array, table bases equal to rows bases
        return self.row_bases


class _ArrXMrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=ARR & columns=MR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        return self.counts

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- Only selected from the selection dimension
        return self._counts[:, :, 0]

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        # --- selected and not-selected both contribute to margin (axis=2), both rows
        # --- and columns are retained.
        return np.sum(self._counts, axis=2)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, but use row bases to add
        # --- over the selected/not selected dimension
        return self.row_bases

    @lazyproperty
    def _columns_pruning_base(self):
        """1D bool np.ndarray of the sum of the row-bases for cells in a row.

        Used to compute the columns-pruning-mask; not meaningful on its own.
        """
        # --- Because column is MR we need to override. We want to sum over the
        # --- selection dimension (dim=2) & the rows (dim=0)
        return np.sum(self._counts, axis=(0, 2))


class _CatXArrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=CAT & columns=ARR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        return np.broadcast_to(self.columns_base, self.counts.shape)

    @lazyproperty
    def columns_base(self):
        """Optional 1D np.float64 ndarray of column-wise base for each column.

        This marginal is used as the (un)weighted N row that is shown (typically)
        below other measures.
        """
        # --- Avaialable because row is CAT, equal to the sum across rows dimension.
        return np.sum(self._counts, axis=0)

    @lazyproperty
    def columns_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each column.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.columns_table_base` for details.
        """
        # --- Available because row is CAT, equal to columns_base because column is ARR
        return self.columns_base

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- No MR, so counts are already in correct shape
        return self._counts

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, bases are equal to counts
        return self.counts

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- Can't sum across columns array, table bases equal to column bases
        return self.column_bases


class _CatXCatCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=CAT & columns=CAT dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        return np.broadcast_to(self.columns_base, self.counts.shape)

    @lazyproperty
    def columns_base(self):
        """Optional 1D np.float64 ndarray of column-wise base for each column.

        This marginal is used as the (un)weighted N row that is shown (typically)
        below other measures.
        """
        # --- Avaialable because row is CAT, equal to the sum across rows dimension.
        return np.sum(self._counts, axis=0)

    @lazyproperty
    def columns_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each column.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.columns_table_base` for details.
        """
        # --- Available because rows are CAT, equal to a repeat of the scalar table base
        # --- because both dimensions are CAT.
        return np.repeat(self.table_base, self.counts.shape[1])

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- No MR, so counts are already in correct shape
        return self._counts

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        return np.broadcast_to(self.rows_base[:, None], self.counts.shape)

    @lazyproperty
    def rows_base(self):
        """Optional 1D np.float64 ndarray of row-wise base for each row.

        This marginal is used as the (un)weighted N column that is shown alongside other
        measures.
        """
        # --- Avaialable because column is CAT, equal to the sum across cols dimension.
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def rows_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each row.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.rows_table_base` for details.
        """
        # --- Available because columns are CAT, equal to a repeat of the scalar table
        # --- base because both dimensions are CAT.
        return np.repeat(self.table_base, self.counts.shape[0])

    @lazyproperty
    def table_base(self):
        """Optional float of the table-wise base for the whole table."""
        # --- Available because both dimensions are CAT, equal to sum of counts
        return np.sum(self._counts)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- Scalar table_base, broadcast it to the correct shape
        return np.broadcast_to(self.table_base, self.counts.shape)


class _CatXMrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=CAT & columns=MR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        return np.broadcast_to(self.columns_base, self.counts.shape)

    @lazyproperty
    def columns_base(self):
        """Optional 1D np.float64 ndarray of column-wise base for each column.

        This marginal is used as the (un)weighted N row that is shown (typically)
        below other measures.
        """
        # --- Avaialable because row is CAT, equal to the sum across rows dimension.
        return np.sum(self.counts, axis=0)

    @lazyproperty
    def columns_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each column.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.columns_table_base` for details.
        """
        return np.sum(self._counts, axis=(0, 2))

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- No MR, so counts are already in correct shape
        return self._counts[:, :, 0]

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        # --- selected and not-selected both contribute to margin (axis=2), both rows
        # --- and columns are retained.
        return np.sum(self._counts, axis=2)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- weighted-counts is (rows, cols, selected/not) so axis 1 is preserved to
        # --- provide a distinct value for each MR subvar.
        return np.broadcast_to(self.columns_table_base, self.counts.shape)

    @lazyproperty
    def _columns_pruning_base(self):
        """1D bool np.ndarray of the sum of the row-bases for cells in a row.

        Used to compute the columns-pruning-mask; not meaningful on its own.
        """
        # --- Because column is MR we need to override. We want to sum over the
        # --- selection dimension (dim=2) & the rows (dim=0)
        return np.sum(self._counts, axis=(0, 2))


class _MrXArrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=MR & columns=ARR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        # --- selected and not-selected both contribute to margin (axis=2), both rows
        # --- and columns are retained.
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- Only selected from the selection dimension
        return self._counts[:, 0, :]

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        return self.counts

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- No addition across subvariables possible, but use column bases to add
        # --- over the selected/not selected dimension
        return self.column_bases

    @lazyproperty
    def _rows_pruning_base(self):
        """1D bool np.ndarray of the sum of the column-bases for cells in a column.

        Used to compute the rows-pruning-mask; not meaningful on its own.
        """
        # --- Because row is MR we need to override. We want to sum over the
        # --- selection dimension (dim=1) & the columns (dim=1)
        return np.sum(self._counts, axis=(1, 2))


class _MrXCatCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=MR & columns=CAT dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        # --- selected and not-selected both contribute to margin (axis=1), both rows
        # --- and columns are retained.
        return np.sum(self._counts, axis=1)

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each valid matrix cell."""
        # --- Only selected from the selection dimension
        return self._counts[:, 0, :]

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        return np.broadcast_to(self.rows_base[:, None], self.counts.shape)

    @lazyproperty
    def rows_base(self):
        """Optional 1D np.float64 ndarray of row-wise base for each row.

        This marginal is used as the (un)weighted N column that is shown alongside other
        measures.
        """
        # --- Avaialable because column is CAT, equal to the sum across cols dimension.
        return np.sum(self.counts, axis=1)

    @lazyproperty
    def rows_table_base(self):
        """Optional 1D np.float64 ndarray of table-wise base for each row.

        This marginal is used as the denominator of the margin-table-proportions. The
        name is a mouthful, see `_BaseCubeCounts.rows_table_base` for details.
        """
        # --- Since the rows-dimension is MR, each row has a distinct table base, since
        # --- not all of the multiple responses were necessarily offered to all
        # --- respondents. The table-base for each row indicates the weighted number
        # --- of respondents who were offered that option.
        return np.sum(self._counts, axis=(1, 2))

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- Broadcast the rows_table_base to the correct shape
        return np.broadcast_to(self.rows_table_base[:, None], self.counts.shape)

    @lazyproperty
    def _rows_pruning_base(self):
        """1D bool np.ndarray of the sum of the column-bases for cells in a column.

        Used to compute the rows-pruning-mask; not meaningful on its own.
        """
        # --- Because row is MR we need to override. We want to sum over the
        # --- selection dimension (dim=1) & the columns (dim=1)
        return np.sum(self._counts, axis=(1, 2))


class _MrXMrCubeCounts(_BaseCubeCounts):
    """Counts cube-measure for a slice with rows=MR & columns=MR dimensions"""

    @lazyproperty
    def column_bases(self):
        """2D np.float64 ndarray of column-wise bases for each matrix cell."""
        # --- only column-selected counts contribute ([:, :, :, 0]), row-selected and
        # --- not-selected are summed (axis=1), rows and columns are retained.
        return np.sum(self._counts[:, :, :, 0], axis=1)

    @lazyproperty
    def counts(self):
        """2D np.float64 ndarray of the count for each matrix cell."""
        # --- Only selected from the selection dimension
        return self._counts[:, 0, :, 0]

    @lazyproperty
    def row_bases(self):
        """2D np.float64 ndarray of row-wise bases for each matrix cell."""
        # --- only selecteds in rows contribute ([:, 0, :, :]), selected and not from
        # --- columns both contribute (axis=2 after rows sel/not axis is collapsed)
        return np.sum(self._counts[:, 0, :, :], axis=2)

    @lazyproperty
    def table_bases(self):
        """2D np.float64 ndarray of table-wise bases for each matrix cell."""
        # --- Reduce second and fourth axes (the two MR_CAT dimensions) with sum()
        # --- producing 2D (nrows, ncols). This sums the (selected, selected),
        # --- (selected, not), (not, selected) and (not, not) cells of the subtable for
        # --- each matrix cell.
        return np.sum(self._counts, axis=(1, 3))

    @lazyproperty
    def _columns_pruning_base(self):
        """1D bool np.ndarray of the sum of the row-bases for cells in a row.

        Used to compute the columns-pruning-mask; not meaningful on its own.
        """
        return np.sum(self._counts[:, :, :, 0], axis=(0, 1))

    @lazyproperty
    def _rows_pruning_base(self):
        """1D bool np.ndarray of the sum of the column-bases for cells in a column.

        Used to compute the rows-pruning-mask; not meaningful on its own.
        """
        return np.sum(self._counts[:, 0, :, :], axis=(1, 2))


# === MEANS ===


class _BaseCubeMeans(_BaseCubeMeasure):
    """Base class for mean cube-measure variants."""

    def __init__(self, dimensions, means):
        super(_BaseCubeMeans, self).__init__(dimensions)
        self._means = means

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeMeans subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-means measure.
        """
        if cube.means is None:
            raise ValueError("cube-result does not contain cube-means measure")
        dimension_types = cube.dimension_types[-2:]
        CubeMeansCls = (
            _MrXMrCubeMeans
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeMeans
            if dimension_types[0] == DT.MR
            else _CatXMrCubeMeans
            if dimension_types[1] == DT.MR
            else _CatXCatCubeMeans
        )
        return CubeMeansCls(
            dimensions, cube.means[cls._slice_idx_expr(cube, slice_idx)]
        )

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of cube means."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.means`"
        )


class _CatXCatCubeMeans(_BaseCubeMeans):
    """Means cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means


class _CatXMrCubeMeans(_BaseCubeMeans):
    """Means cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means[:, :, 0]


class _MrXCatCubeMeans(_BaseCubeMeans):
    """Means cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        return self._means[:, 0, :]


class _MrXMrCubeMeans(_BaseCubeMeans):
    """Means cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def means(self):
        """2D np.float64 ndarray of means for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._means[:, 0, :, 0]


# === OVERLAPS ===


class _BaseCubeOverlaps(_BaseCubeMeasure):
    """Base class for overlap cube-measure variants."""

    def __init__(self, dimensions, overlaps, valid_overlaps):
        super(_BaseCubeOverlaps, self).__init__(dimensions)
        self._overlaps = overlaps
        self._valid_overlaps = valid_overlaps

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeOverlaps subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-overlaps measure
        or if it doesn't include valid-cube-overlaps measure.
        """
        if cube.overlaps is None:
            raise ValueError(
                "cube-result does not contain cube-overlaps measure"
            )  # pragma: no cover
        if cube.valid_overlaps is None:
            raise ValueError(
                "cube-result does not contain cube-valid-overlaps measure"
            )  # pragma: no cover

        dimension_types = tuple(d.dimension_type for d in dimensions)
        idx_expr = cls._slice_idx_expr(cube, slice_idx)
        args = (dimensions, cube.overlaps[idx_expr], cube.valid_overlaps[idx_expr])
        return (
            _MrXMrOverlaps(*args)
            if dimension_types == (DT.MR, DT.MR)
            else _CatXMrOverlaps(*args)
        )

    @lazyproperty
    def overlaps(self):
        """3D np.float64 ndarray of cube overlaps."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.overlaps`"
        )


class _CatXMrOverlaps(_BaseCubeOverlaps):
    """Overlaps cube-measure for a NOT_MR_X_MR slice."""

    @lazyproperty
    def tile_repetitions(self):
        """tuple of number of repetitions of selected base matrix."""
        return (self._overlaps.shape[0], 1, 1)

    @lazyproperty
    def selected_bases(self):
        """3D np.float64 ndarray of selected overlaps between MR subvariables, per cat.

        For a CAT x MR matrix, the overlaps are calculated for each category, and then
        for each subvariables pair for that category (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is CAT x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the result
        of the `cube_overlap` measure, and representing the "pairing" with each subvar
        of the previous MR_SUBVAR dimension.

        From this shape, we only need the "Selected" part of the MR_SEL dimension, so
        we need to select the 0th element along the 2nd axis [:, :, 0]. But we also need
        to add all of the categories together, since that's what's used for the base
        value in all of the significance calculations. We then tile these bases, since
        they're the same for each row (all the categories added together). The tiling is
        done so that the API of this class can be uniform for all users.
        """
        return np.tile(np.sum(self._overlaps[:, :, 0], axis=0), self.tile_repetitions)

    @lazyproperty
    def valid_bases(self):
        """3D np.float64 ndarray of valid overlaps between MR subvariables, per cat.

        For a CAT x MR matrix, the overlaps are calculated for each category, and then
        for each subvariables pair for that category (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is CAT x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the result
        of the `cube_overlap` measure, and representing the "pairing" with each subvar
        of the previous MR_SUBVAR dimension.

        From this shape, we only need the "Selected" + "Other" part of the MR_SEL
        dimension (i.e. all except missing), so we need to add the 0th and the 1st
        element along the 2nd axis sum([:, :, 0:2]).
        """
        valids = np.sum(self._valid_overlaps[:, :, 0:2], axis=2)
        return np.tile(np.sum(valids, axis=0), self.tile_repetitions)


class _MrXMrOverlaps(_BaseCubeOverlaps):
    """Overlaps cube-measure for a MR_X_MR slice."""

    @lazyproperty
    def selected_bases(self):
        """3D np.float64 ndarray of selected overlaps bases between MR subvariables.

        For a MR x MR slice, the overlaps are calculated for each MR subvar row, and
        then for each subvariables pair for that row (which will produce a square
        matrix for each category). So the output shape that we get back from the
        database is MR_SUBVAR x MR_SEL x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one
        being the result of the `cube_overlap` measure, and representing the "pairing"
        with each subvar of the last MR_SUBVAR dimension.

        From this shape, we only need the "Selected" part of the last MR_SEL dimension,
        but we need both "Selected" and "Other" from the first MR_SEL dimension. We
        therefore need to select the 0th element along the 3rd axis, but we need to sum
        the 0th and 1st elements along the 1st axis (first MR_SEL dimension). The sum
        of the selected and other counts of the first MR_SEL is what's always used to
        represent bases of such crosstabs (we cannot simply ignore the other counts).
        """
        return np.sum(self._overlaps[:, 0:2, :, 0], axis=1)

    @lazyproperty
    def valid_bases(self):
        """3D np.float64 ndarray of valid overlaps between MR subvariables, per row.

        For a MR x MR slice, the overlaps are calculated for each row, and then
        for each subvariables pair for that row (which will produce a square matrix for
        each row). So the output shape that we get back from the database is
        MR_SUBVAR x MR_SEL x MR_SUBVAR x MR_SEL x MR_SUBVAR (the last one being the
        result of the `cube_overlap` measure, and representing the "pairing" with each
        subvar of the last MR_SUBVAR dimension.

        From this shape, we only need the "Selected" + "Other" part of both MR_SEL
        dimensions (i.e. all except missing), so we need to add the 0th and the 1st
        element along the 1st and 3rd axes sum([:, 0:2, :, 0:2], axis=(1, 3)).
        """
        return np.sum(self._valid_overlaps[:, 0:2, :, 0:2], axis=(1, 3))


# === STD DEV ===


class _BaseCubeStdDev(_BaseCubeMeasure):
    """Base class for stddev cube-measure variants."""

    def __init__(self, dimensions, stddev):
        super(_BaseCubeStdDev, self).__init__(dimensions)
        self._stddev = stddev

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeStdDev subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-stddev measure.
        """
        if cube.stddev is None:
            raise ValueError("cube-result does not contain cube-stddev measure")
        dimension_types = cube.dimension_types[-2:]
        CubeSumsCls = (
            _MrXMrCubeStdDev
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeStdDev
            if dimension_types[0] == DT.MR
            else _CatXMrCubeStdDev
            if dimension_types[1] == DT.MR
            else _CatXCatCubeStdDev
        )
        return CubeSumsCls(
            dimensions, cube.stddev[cls._slice_idx_expr(cube, slice_idx)]
        )

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of cube stddev."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.stddev`"
        )


class _CatXCatCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev


class _CatXMrCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev[:, :, 0]


class _MrXCatCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        return self._stddev[:, 0, :]


class _MrXMrCubeStdDev(_BaseCubeStdDev):
    """StdDev cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def stddev(self):
        """2D np.float64 ndarray of stddev for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._stddev[:, 0, :, 0]


# === SUMS ===


class _BaseCubeSums(_BaseCubeMeasure):
    """Base class for sum cube-measure variants."""

    def __init__(self, dimensions, sums):
        super(_BaseCubeSums, self).__init__(dimensions)
        self._sums = sums

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseCubeSums subclass instance appropriate to `cube`.

        Raises `ValueError` if the cube-result does not include a cube-sum measure.
        """
        if cube.sums is None:
            raise ValueError("cube-result does not contain cube-sum measure")
        dimension_types = cube.dimension_types[-2:]
        CubeSumsCls = (
            _MrXMrCubeSums
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatCubeSums
            if dimension_types[0] == DT.MR
            else _CatXMrCubeSums
            if dimension_types[1] == DT.MR
            else _CatXCatCubeSums
        )
        return CubeSumsCls(dimensions, cube.sums[cls._slice_idx_expr(cube, slice_idx)])

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of cube sum."""
        raise NotImplementedError(  # pragma: no cover
            f"`{type(self).__name__}` must implement `.sum`"
        )


class _CatXCatCubeSums(_BaseCubeSums):
    """Sums cube-measure for a slice with no MR dimensions."""

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums


class _CatXMrCubeSums(_BaseCubeSums):
    """Sums cube-measure for a NOT_MR_X_MR slice.

    Note that the rows-dimensions need not actually be CAT.
    """

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums[:, :, 0]


class _MrXCatCubeSums(_BaseCubeSums):
    """Sums cube-measure for an MR_X_NOT_MR slice.

    Note that the columns-dimension need not actually be CAT.
    """

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        return self._sums[:, 0, :]


class _MrXMrCubeSums(_BaseCubeSums):
    """Sums cube-measure for an MR_X_MR slice."""

    @lazyproperty
    def sums(self):
        """2D np.float64 ndarray of sum for each valid matrix cell."""
        # --- indexing is: all-rows, sel-only, all-cols, sel-only ---
        return self._sums[:, 0, :, 0]


# === (WEIGHTED) UNCONDITIONAL COUNTS ===


class _BaseUnconditionalCubeCounts(_BaseCubeMeasure):
    """Base class for unconditional-(weighted)-count cube-measure variants.

    Unconditional counts are the weighted counts including the missing (invalid) rows/
    columns. This cube measure is used to calculate the column-index. The name does
    not include "weighted" because we do not (currently) ever need the unweighted
    version of the unconditional counts, but they could in theory be calculated.
    """

    def __init__(
        self,
        dimensions,
        counts_with_missings,
    ):
        super(_BaseUnconditionalCubeCounts, self).__init__(dimensions)
        self._counts_with_missings = counts_with_missings

    @classmethod
    def factory(cls, cube, dimensions, slice_idx):
        """Return _BaseWeightedCounts subclass instance appropriate to `cube`."""
        dimension_types = cube.dimension_types[-2:]
        counts_with_missings = cube.counts_with_missings

        # --- TODO: This probably needs Array type subclasses like the other counts
        # --- cube measures have, because we shouldn't be adding across subvariables
        # --- I don't think we ever actually use this, though, and would also need to
        # --- choose between weighted_counts and valid_weighted_counts
        UnconditionalCubeCountsCls = (
            _MrXMrUnconditionalCubeCounts
            if dimension_types == (DT.MR, DT.MR)
            else _MrXCatUnconditionalCubeCounts
            if dimension_types[0] == DT.MR
            else _CatXMrUnconditionalCubeCounts
            if dimension_types[1] == DT.MR
            else _CatXCatUnconditionalCubeCounts
        )
        return UnconditionalCubeCountsCls(
            dimensions,
            counts_with_missings[cls._slice_idx_expr(cube, slice_idx)],
        )

    @lazyproperty
    def _valid_row_idxs(self):
        """ndarray-style index for only valid (non-missing) rows.

        Suitable for indexing a raw measure array to include only valid rows.
        """
        return np.ix_(self._dimensions[-2].valid_elements.element_idxs)


class _CatXCatUnconditionalCubeCounts(_BaseUnconditionalCubeCounts):
    """Unconditional-counts cube-measure for a slice with no MR dimensions.

    Unconditional counts are the counts including the missing (invalid) rows/
    columns. This cube measure is used to calculate the column-index.
    """

    @lazyproperty
    def baseline(self):
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


class _CatXMrUnconditionalCubeCounts(_BaseUnconditionalCubeCounts):
    """Unconditional-counts cube-measure for a NOT_MR_X_MR slice.

    Unconditional counts are the counts including the missing (invalid) rows/
    columns. This cube measure is used to calculate the column-index.
    """

    @lazyproperty
    def baseline(self):
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


class _MrXCatUnconditionalCubeCounts(_BaseUnconditionalCubeCounts):
    """Unconditional-counts cube-measure for an MR_X_NOT_MR slice.

    Unconditional counts are the counts including the missing (invalid) rows/
    columns. This cube measure is used to calculate the column-index.
    """

    @lazyproperty
    def baseline(self):
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


class _MrXMrUnconditionalCubeCounts(_BaseUnconditionalCubeCounts):
    """Weighted-counts cube-measure for an MR_X_MR slice.

    Unconditional counts are the counts including the missing (invalid) rows/
    columns. This cube measure is used to calculate the column-index.
    """

    @lazyproperty
    def baseline(self):
        """2D np.float64 ndarray of baseline value for each matrix cell.

        The shape is (nrows, ncols) and all values in a given row are the same. So
        really there are only nrows distinct baseline values, but the returned shape
        makes calculating column-index in a general way more convenient.
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
        # ---    MR_CAT dimension (rows MR). ("counts" for MR are "selected" counts).
        # --- Produces a 2D (nrows, ncols) array.
        uncond_row_margin = np.sum(self._counts_with_missings[:, 0:2], axis=3)[:, 0]
        # --- Unconditional table margin is also 2D (nrows, ncols) but the values for
        # --- all columns in a row have the same value; basically each row has
        # --- a distinct table margin.
        uncond_table_margin = np.sum(self._counts_with_missings[:, 0:2], axis=(1, 3))
        # --- baseline is produced by dividing uncond_row_margin by uncond_table_margin.
        return uncond_row_margin / uncond_table_margin
