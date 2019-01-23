# encoding: utf-8

"""CubeSlice class."""

from __future__ import division

import warnings
from functools import partial

import numpy as np
from tabulate import tabulate
from scipy.stats import norm
from scipy.stats.contingency import expected_freq

from cr.cube.enum import DIMENSION_TYPE as DT
from cr.cube.measures.scale_means import ScaleMeans
from cr.cube.measures.pairwise_pvalues import PairwisePvalues
from cr.cube.util import compress_pruned, lazyproperty, memoize


class CubeSlice(object):
    """Two-dimensional projection of a cube.

    The CubeSlice object is used to uniformly represent any cube as one or
    more 2D cubes. For 1D cubes, this is achieved by inflating. For 3D cubes,
    this is achieved by slicing. CubeSlice objects for a cube are accessed
    using the :meth:`.CrunchCube.get_slices` method.
    """

    row_dim_ind = 0

    def __init__(self, cube, index, ca_as_0th=False):

        if ca_as_0th and cube.dim_types[0] != DT.CA_SUBVAR:
            msg = (
                "Cannot set CA as 0th for cube that "
                "does not have CA items as the 0th dimension."
            )
            raise ValueError(msg)

        self._cube = cube
        self._index = index
        self.ca_as_0th = ca_as_0th

    def __getattr__(self, attr):
        # --Invoked when self.{attr} raises AttributeError, i.e. when
        # --CubeSlice does not define a method or property and it needs to be
        # --delegated to the underlying Cube instance
        cube_attr = getattr(self._cube, attr)

        # API Method calls
        if callable(cube_attr):
            return partial(self._call_cube_method, attr)

        # API properties
        get_only_last_two = (
            self._cube.ndim == 3 and hasattr(cube_attr, "__len__") and attr != "name"
        )
        if get_only_last_two:
            return cube_attr[-2:]

        # ---otherwise, the property value is the same for cube or slice---
        return cube_attr

    def __repr__(self):
        """Provide text representation suitable for working at console.

        Falls back to a default repr on exception, such as might occur in
        unit tests where object need not otherwise be provided with all
        instance variable values.
        """
        try:
            name = self.table_name if self.table_name else self.name
            dimensionality = " x ".join(dt.name for dt in self.dim_types)
            dim_names = " x ".join(d.name for d in self.dimensions)
            labels = self.labels()
            headers = labels[1] if len(labels) > 1 else ["N"]
            tbl = (
                self.as_array()
                if len(self.as_array().shape) > 1
                else self.as_array()[:, None]
            )
            body = tabulate(
                [[lbl] + lst for lbl, lst in zip(labels[0], tbl.tolist())],
                headers=headers,
            )
            return "%s(name='%s', dim_types='%s', dims='%s')\n%s" % (
                type(self).__name__,
                name,
                dimensionality,
                dim_names,
                body,
            )
        except Exception:
            return super(CubeSlice, self).__repr__()

    @lazyproperty
    def ca_main_axis(self):
        """For univariate CA, the main axis is the categorical axis"""
        try:
            ca_ind = self.dim_types.index(DT.CA_SUBVAR)
            return 1 - ca_ind
        except ValueError:
            return None

    @lazyproperty
    def col_dim_ind(self):
        """Return 1 if not categorical array as 0th, 0 otherwise."""
        return 1 if not self.ca_as_0th else 0

    @lazyproperty
    def can_compare_pairwise(self):
        return self.dim_types == (DT.CAT, DT.CAT)

    @lazyproperty
    def dim_types(self):
        """Tuple of DIMENSION_TYPE member for each dimension of slice."""
        return self._cube.dim_types[-2:]

    @memoize
    def get_shape(self, prune=False):
        """Tuple of array dimensions' lengths.

        It returns a tuple of ints, each representing the length of a cube
        dimension, in the order those dimensions appear in the cube.
        Pruning is supported. Dimensions that get reduced to a single element
        (e.g. due to pruning) are removed from the returning shape, thus
        allowing for the differentiation between true 2D cubes (over which
        statistical testing can be performed) and essentially
        1D cubes (over which it can't).

        Usage:

        >>> shape = get_shape()
        >>> pruned_shape = get_shape(prune=True)
        """
        if not prune:
            return self.as_array().shape

        shape = compress_pruned(self.as_array(prune=True)).shape
        # Eliminate dimensions that get reduced to 1
        # (e.g. single element categoricals)
        return tuple(n for n in shape if n > 1)

    @lazyproperty
    def has_ca(self):
        """Check if the cube slice has the CA dimension.

        This is used to distinguish between slices that are considered 'normal'
        (like CAT x CAT), that might be a part of the 3D cube that has 0th dim
        as the CA items (subvars). In such a case, we still need to process
        the slices 'normally', and not address the CA items constraints.
        """
        return DT.CA_SUBVAR in self.dim_types

    @lazyproperty
    def has_mr(self):
        """True if the slice has MR dimension.

        This property needs to be overridden, because we don't care about the
        0th dimension (and if it's an MR) in the case of a 3D cube.
        """
        return DT.MR in self.dim_types

    def index_table(self, axis=None, baseline=None, prune=False):
        """Return index percentages for a given axis and baseline.

        The index values represent the difference of the percentages to the
        corresponding baseline values. The baseline values are the univariate
        percentages of the corresponding variable.
        """
        proportions = self.proportions(axis=axis)
        baseline = (
            baseline if baseline is not None else self._prepare_index_baseline(axis)
        )

        # Fix the shape to enable correct broadcasting
        if (
            axis == 0
            and len(baseline.shape) <= 1
            and self.ndim == len(self.get_shape())
        ):
            baseline = baseline[:, None]

        indexes = proportions / baseline * 100

        return self._apply_pruning_mask(indexes) if prune else indexes

    @lazyproperty
    def is_double_mr(self):
        """This has to be overridden from cr.cube.

        If the underlying cube is 3D, the 0th dimension must not be taken into
        account, since it's only the tabs dimension, and mustn't affect the
        properties of the slices.
        """
        return self.dim_types == (DT.MR, DT.MR)

    def labels(self, hs_dims=None, prune=False):
        """Get labels for the cube slice, and perform pruning by slice."""
        if self.ca_as_0th:
            labels = self._cube.labels(include_transforms_for_dims=hs_dims)[1:]
        else:
            labels = self._cube.labels(include_transforms_for_dims=hs_dims)[-2:]

        if not prune:
            return labels

        def prune_dimension_labels(labels, prune_indices):
            """Get pruned labels for single dimension, besed on prune inds."""
            labels = [label for label, prune in zip(labels, prune_indices) if not prune]
            return labels

        labels = [
            prune_dimension_labels(dim_labels, dim_prune_inds)
            for dim_labels, dim_prune_inds in zip(
                labels, self._prune_indices(transforms=hs_dims)
            )
        ]
        return labels

    @lazyproperty
    def mr_dim_ind(self):
        """Get the correct index of the MR dimension in the cube slice."""
        mr_dim_ind = self._cube.mr_dim_ind
        if self._cube.ndim == 3:
            if isinstance(mr_dim_ind, int):
                if mr_dim_ind == 0:
                    # If only the 0th dimension of a 3D is an MR, the sliced
                    # don't actuall have the MR... Thus return None.
                    return None
                return mr_dim_ind - 1
            elif isinstance(mr_dim_ind, tuple):
                # If MR dimension index is a tuple, that means that the cube
                # (only a 3D one if it reached this path) has 2 MR dimensions.
                # If any of those is 0 ind dimension we don't need to include
                # in the slice dimension (because the slice doesn't see the tab
                # that it's on). If it's 1st and 2nd dimension, then subtract 1
                # from those, and present them as 0th and 1st dimension of the
                # slice. This can happend e.g. in a CAT x MR x MR cube (which
                # renders MR x MR slices).
                mr_dim_ind = tuple(i - 1 for i in mr_dim_ind if i)
                return mr_dim_ind if len(mr_dim_ind) > 1 else mr_dim_ind[0]

        return mr_dim_ind

    @lazyproperty
    def ndim(self):
        """Number of slice dimensions

        Returns 2 if the origin cube has 3 or 2 dimensions.  Returns 1 if the
        cube (and the slice) has 1 dimension.  Returns 0 if the cube doesn't
        have any dimensions.
        """
        return min(self._cube.ndim, 2)

    def scale_means(self, hs_dims=None, prune=False):
        """Return list of column and row scaled means for this slice.

        If a row/col doesn't have numerical values, return None for the
        corresponding dimension. If a slice only has 1D, return only the column
        scaled mean (as numpy array). If both row and col scaled means are
        present, return them as two numpy arrays inside of a list.
        """
        scale_means = self._cube.scale_means(hs_dims, prune)
        if self.ca_as_0th:
            return [scale_means[0][-1][self._index]]
        return self._cube.scale_means(hs_dims, prune)[self._index]

    @memoize
    def scale_means_margin(self, axis):
        """Get scale means margin for 2D slice.

        This value represents the scale mean of a single variable that
        constitutes a 2D slice. There's one for each axis, if there are
        numerical values on the corresponding (opposite) dimension. The
        numerical values are filtered by non-missing criterium of the
        opposite dimension.
        """
        return ScaleMeans(self).margin(axis)

    @lazyproperty
    def shape(self):
        """Tuple of array dimensions' lengths.

        It returns a tuple of ints, each representing the length of a cube
        dimension, in the order those dimensions appear in the cube.

        This property is deprecated, use 'get_shape' instead. Pruning is not
        supported (supported in 'get_shape').
        """
        deprecation_msg = "Deprecated. Use `get_shape` instead."
        warnings.warn(deprecation_msg, DeprecationWarning)
        return self.get_shape()

    @lazyproperty
    def table_name(self):
        """Get slice name.

        In case of 2D return cube name. In case of 3D, return the combination
        of the cube name with the label of the corresponding slice
        (nth label of the 0th dimension).
        """
        if self._cube.ndim < 3 and not self.ca_as_0th:
            return None

        title = self._cube.name
        table_name = self._cube.labels()[0][self._index]
        return "%s: %s" % (title, table_name)

    def pairwise_pvals(self, axis=0):
        """Return square symmetric matrix of pairwise column-comparison p-values.

        Square, symmetric matrix along *axis* of pairwise p-values for the
        null hypothesis that col[i] = col[j] for each pair of columns.

        *axis* (int): axis along which to perform comparison. Only columns (0)
        are implemented currently.
        """
        if axis != 0:
            raise NotImplementedError("Pairwise comparison only implemented for colums")
        return PairwisePvalues(self, axis=axis).values

    def pvals(self, weighted=True, prune=False, hs_dims=None):
        """Return 2D ndarray with calculated P values

        This function calculates statistically significant cells for
        categorical contingency tables under the null hypothesis that the
        row and column variables are independent (uncorrelated).
        The values are calculated for 2D tables only.

        :param weighted: Use weighted counts for zscores
        :param prune: Prune based on unweighted counts
        :param hs_dims: Include headers and subtotals (as NaN values)
        :returns: 2 or 3 Dimensional ndarray, representing the p-values for each
                  cell of the table-like representation of the crunch cube.
        """
        stats = self.zscore(weighted=weighted, prune=prune, hs_dims=hs_dims)
        pvals = 2 * (1 - norm.cdf(np.abs(stats)))

        return self._apply_pruning_mask(pvals, hs_dims) if prune else pvals

    def zscore(self, weighted=True, prune=False, hs_dims=None):
        """Return ndarray with slices's standardized residuals (Z-scores).

        (Only applicable to a 2D contingency tables.) The Z-score or
        standardized residual is the difference between observed and expected
        cell counts if row and column variables were independent divided
        by the residual cell variance. They are assumed to come from a N(0,1)
        or standard Normal distribution, and can show which cells deviate from
        the null hypothesis that the row and column variables are uncorrelated.

        See also *pairwise_chisq*, *pairwise_pvals* for a pairwise column-
        or row-based test of statistical significance.

        :param weighted: Use weighted counts for zscores
        :param prune: Prune based on unweighted counts
        :param hs_dims: Include headers and subtotals (as NaN values)
        :returns zscore: ndarray representing cell standardized residuals (Z)
        """
        counts = self.as_array(weighted=weighted)
        total = self.margin(weighted=weighted)
        colsum = self.margin(axis=0, weighted=weighted)
        rowsum = self.margin(axis=1, weighted=weighted)
        zscore = self._calculate_std_res(counts, total, colsum, rowsum)

        if hs_dims:
            zscore = self._intersperse_hs_in_std_res(hs_dims, zscore)

        if prune:
            return self._apply_pruning_mask(zscore, hs_dims)

        return zscore

    def _apply_pruning_mask(self, res, hs_dims=None):
        array = self.as_array(prune=True, include_transforms_for_dims=hs_dims)

        if not isinstance(array, np.ma.core.MaskedArray):
            return res

        return np.ma.masked_array(res, mask=array.mask)

    def _array_type_std_res(self, counts, total, colsum, rowsum):
        """Return ndarray containing standard residuals for array values.

        The shape of the return value is the same as that of *counts*.
        Array variables require special processing because of the
        underlying math. Essentially, it boils down to the fact that the
        variable dimensions are mutually independent, and standard residuals
        are calculated for each of them separately, and then stacked together
        in the resulting array.
        """
        if self.mr_dim_ind == 0:
            # --This is a special case where broadcasting cannot be
            # --automatically done. We need to "inflate" the single dimensional
            # --ndarrays, to be able to treat them as "columns" (essentially a
            # --Nx1 ndarray). This is needed for subsequent multiplication
            # --that needs to happen column wise (rowsum * colsum) / total.
            total = total[:, np.newaxis]
            rowsum = rowsum[:, np.newaxis]

        expected_counts = rowsum * colsum / total
        variance = rowsum * colsum * (total - rowsum) * (total - colsum) / total ** 3
        return (counts - expected_counts) / np.sqrt(variance)

    def _calculate_std_res(self, counts, total, colsum, rowsum):
        """Return ndarray containing standard residuals.

        The shape of the return value is the same as that of *counts*.
        """
        if set(self.dim_types) & DT.ARRAY_TYPES:  # ---has-mr-or-ca---
            return self._array_type_std_res(counts, total, colsum, rowsum)
        return self._scalar_type_std_res(counts, total, colsum, rowsum)

    def _call_cube_method(self, method, *args, **kwargs):
        kwargs = self._update_args(kwargs)
        result = getattr(self._cube, method)(*args, **kwargs)
        if method == "inserted_hs_indices":
            if not self.ca_as_0th:
                result = result[-2:]
            return result
        return self._update_result(result)

    def _intersperse_hs_in_std_res(self, hs_dims, res):
        for dim, inds in enumerate(self.inserted_hs_indices()):
            for i in inds:
                if dim not in hs_dims:
                    continue
                res = np.insert(res, i, np.nan, axis=(dim - self.ndim))
        return res

    def _prepare_index_baseline(self, axis):
        # First get the margin of the opposite direction of the index axis.
        # We need this in order to end up with the right shape of the
        # numerator vs denominator.
        baseline = self.margin(axis=(1 - axis), include_missing=True)
        if len(baseline.shape) <= 1:
            # If any dimension gets flattened out, due to having a single
            # element, re-inflate it
            baseline = baseline[None, :]
        slice_ = [slice(None)]
        total_axis = None
        if isinstance(self.mr_dim_ind, tuple):
            if self.get_shape()[0] == 1 and axis == 0:
                total_axis = axis
                slice_ = [0]
            elif self.get_shape()[0] == 1 and axis == 1:
                total_axis = 1
                slice_ = [slice(None), 0]
            else:
                total_axis = axis + 1
                slice_ += [slice(None), 0] if axis == 1 else [0]
        elif self.mr_dim_ind is not None:
            slice_ = [0] if self.mr_dim_ind == 0 and axis != 0 else [slice(None), 0]
            total_axis = axis if self.mr_dim_ind != 0 else 1 - axis

        total = np.sum(baseline, axis=total_axis)
        baseline = baseline[slice_]

        if axis == self.mr_dim_ind:
            return baseline / total
        elif isinstance(self.mr_dim_ind, tuple) and axis in self.mr_dim_ind:
            return baseline / total

        baseline = baseline if len(baseline.shape) <= 1 else baseline[0]
        baseline = baseline / np.sum(baseline)
        return baseline / np.sum(baseline, axis=0)

    def _scalar_type_std_res(self, counts, total, colsum, rowsum):
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

    def _update_args(self, kwargs):
        if self._cube.ndim < 3:
            # If cube is 2D it doesn't actually have slices (itself is a slice).
            # In this case we don't need to convert any arguments, but just
            # pass them to the underlying cube (which is the slice).
            if self.ca_as_0th:
                axis = kwargs.get("axis", False)
                if axis is None:
                    # Special case for CA slices (in multitables). In this case,
                    # we need to calculate a measurement across CA categories
                    # dimension (and not across items, because it's not
                    # allowed). The value for the axis parameter of None, would
                    # incur the items dimension, and we don't want that.
                    kwargs["axis"] = 1
            return kwargs

        # Handling API methods that include 'axis' parameter

        axis = kwargs.get("axis")
        # Expected usage of the 'axis' parameter from CubeSlice is 0, 1, or
        # None. CrunchCube handles all other logic. The only 'smart' thing
        # about the handling here, is that the axes are increased for 3D cubes.
        # This way the 3Dness is hidden from the user and he still sees 2D
        # crosstabs, with col and row axes (0 and 1), which are transformed to
        # corresponding numbers in case of 3D cubes (namely 1 and 2). In the
        # case of None, we need to analyze across all valid dimensions, and the
        # CrunchCube takes care of that (no need to update axis if it's None).
        # If the user provides a tuple, it's considered that he "knows" what
        # he's doing, and the axis argument is not updated in this case.
        if isinstance(axis, int):
            kwargs["axis"] += 1

        # Handling API methods that include H&S parameter

        # For most cr.cube methods, we use the 'include_transforms_for_dims'
        # parameter name. For some, namely the prune_indices, we use the
        # 'transforms'. These are parameters that tell to the cr.cube "which
        # dimensions to include the H&S for". The only point of this parameter
        # (from the perspective of the cr.exporter) is to exclude the 0th
        # dimension's H&S in the case of 3D cubes.
        hs_dims_key = (
            "transforms" in kwargs
            and "transforms"
            or "hs_dims" in kwargs
            and "hs_dims"
            or "include_transforms_for_dims"
        )
        hs_dims = kwargs.get(hs_dims_key)
        if isinstance(hs_dims, list):
            # Keep the 2D illusion for the user. If a user sees a 2D slice, he
            # still needs to be able to address both dimensions (for which he
            # wants the H&S included) as 0 and 1. Since these are offset by a 0
            # dimension in a 3D case, inside the cr.cube, we need to increase
            # the indexes of the required dims.
            kwargs[hs_dims_key] = [dim + 1 for dim in hs_dims]

        return kwargs

    def _update_result(self, result):
        if self._cube.ndim < 3 and not self.ca_as_0th or len(result) - 1 < self._index:
            return result
        result = result[self._index]
        if isinstance(result, tuple):
            return np.array(result)
        elif not isinstance(result, np.ndarray):
            result = np.array([result])
        return result
