# encoding: utf-8

import warnings

import numpy as np

from ..util import lazyproperty


class SingleSidedMovingAvgSmoother(object):
    """Create and configure smoothing function for one-sided moving average."""

    def __init__(self, partition, **kwargs):
        self._partition = partition
        self._window = kwargs.get("window", 3)
        self._base_measure = kwargs.get("base_measure", None)

    @lazyproperty
    def smoothed_values(self):
        """ -> 1D/2D float64 ndarray of smootehd values including additional nans.

        Given a series of numbers and a fixed subset size, the first element of the
        moving average is obtained by taking the average of the initial fixed subset
        of the number series. Then the subset is modified by `shifting forward` the
        values. A moving average is commonly used with time series data to smooth
        out short-term fluctuations and highlight longer-term trends or cycles.

        The below examples will show 1D and 2D array rolling mean calculations with
        window sizes of 2 and 3, respectively.

                                    [window = 2]
        ----------------------------------------------------------------------------
            x    |   smooth(x)                  x     |        smooth(x)
        ---------+--------------         -------------+------------------------
            1    |    NaN                 1  3  2  3  |   NaN  2.0  2.5  2.5
            2    |    1.5                 2  3  3  2  |   NaN  2.5  3.0  2.5
            3    |    2.5                 3  2  4  4  |   NaN  2.5  3.0  4.0
            4    |    3.5                 4  1  5  1  |   NaN  2.5  3.0  3.0

                                    [window = 3]
        ----------------------------------------------------------------------------
            x    |   smooth(x)                  x     |        smooth(x)
        ---------+--------------         -------------+------------------------
            1    |    NaN                 1  3  2  3  |   NaN  NaN   2.0  2.67
            2    |    NaN                 2  3  3  2  |   NaN  NaN  2.67  2.67
            3    |     2                  3  2  4  4  |   NaN  NaN   3.0  3.33
            4    |     3                  4  1  5  1  |   NaN  NaN  3.33  2.33

        This is performed just taking the average of the last 2 or 3 rows according
        to the window, all the way down the column.
        """
        values = self._values
        if not self._show_smoothing:
            # return original values if smoothing cannot be performed
            return values
        smoothed_values = self._smoothed_values
        # offset between original values and smoothed values
        offset = [values.shape[-1] - smoothed_values.shape[-1]]
        additional_nans = np.full(list(values.shape[:-1]) + offset, np.nan)
        return np.concatenate([additional_nans, smoothed_values], axis=values.ndim - 1)

    @lazyproperty
    def _is_cat_date(self):
        """True for a categorical dimension having date defined on all valid categories.

        Only meaningful when the dimension is known to be categorical
        (has base-type `categorical`).
        """
        categories = self._partition.dimension_dict["type"].get("categories", [])
        if not categories:
            return False
        return all(
            category.get("date")
            for category in categories
            if not category.get("missing", False)
        )

    @lazyproperty
    def _show_smoothing(self):
        """Return True if smoothing is active"""
        # --- no smoothing when smoothing transform is not present ---
        if not self._valid_window:
            warnings.warn(
                "No smoothing performed. Window (value: {}) parameter is not "
                "valid: window must be less than equal to the total period "
                "(value: {}) and positive".format(self._window, self._values.shape[-1]),
                UserWarning,
            )
            return False
        # --- no smoothing when column dimension is not a categorical date ---
        if not self._is_cat_date:
            warnings.warn(
                "No smoothing performed. Column dimension must be a categorical date"
            )
            return False
        return True

    @lazyproperty
    def _smoothed_values(self):
        """ -> np.ndarray, provide smoothed values on the original values.

        In this case the moving average smoother is performed using the np.convolve
        (https://numpy.org/doc/stable/reference/generated/numpy.convolve.html)
        operator that returns the discrete, linear convolution of two one-dimensional
        sequences.
        A moving average is a form of a convolution often used in time series analysis
        to smooth out noise in data by replacing a data point with the average of
        neighboring values in a moving window. A moving average is essentially a
        low-pass filter because it removes short-term fluctuations to highlight a deeper
        underlying trend.
        """
        w = self._window
        values = self._values
        return (
            np.array(tuple(np.convolve(values, np.ones(w), mode="valid") / w))
            if values.ndim == 1
            else np.array(
                [tuple(np.convolve(v, np.ones(w), mode="valid") / w) for v in values]
            )
        )

    @lazyproperty
    def _valid_window(self):
        """ -> bool, the validity of the window parameter.

        Return last dimension size, if window is greater than the the last dimension
        size because we cannot have a moving window grater than the number of elements
        of each column.
        """
        total_period = self._values.shape[-1]
        if self._window > total_period or self._window == 0:
            return False
        return True

    @lazyproperty
    def _values(self):
        """ ndarray, base measure values of the current partition

        The `base_measure` is expressed in the kwargs of the function_spec and used
        to get the values for the partition.
        """
        return np.array(getattr(self._partition, self._base_measure))
