# encoding: utf-8

import warnings

import numpy as np

from cr.cube.enums import DIMENSION_TYPE as DT
from cr.cube.util import lazyproperty


class Smoother(object):
    """Base object class for Smoother variants."""

    @classmethod
    def factory(cls, dimension):
        """Returns appropriate Smoother object according to passed function.

        Raises an exception if the function is different from `one_sided_moving_avg`
        that at the moment is the only one implemented.
        """
        smoothing_dict = dimension.smoothing_dict
        function = smoothing_dict.get("function") or "one_sided_moving_avg"
        if function != "one_sided_moving_avg":
            raise NotImplementedError("Function {} is not available.".format(function))
        return _SingleSidedMovingAvgSmoother(
            smoothing_dict=smoothing_dict,
            dimension_type=dimension.dimension_type,
        )


class _SingleSidedMovingAvgSmoother(object):
    """Create and configure smoothing function for one-sided moving average."""

    def __init__(self, smoothing_dict, dimension_type):
        self._smoothing_dict = smoothing_dict
        self._dimension_type = dimension_type

    def smooth(self, values):
        """1D/2D float64 ndarray of smoothed values including additional nans.

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
        window = self._window
        if not self._can_smooth(values):
            # return original values if smoothing cannot be performed
            return values
        smoothed_values = (
            np.array(tuple(np.convolve(values, np.ones(window), mode="valid") / window))
            if values.ndim == 1
            else np.array(
                [
                    tuple(np.convolve(v, np.ones(window), mode="valid") / window)
                    for v in values
                ]
            )
        )
        # offset between original values and smoothed values
        offset = [values.shape[-1] - smoothed_values.shape[-1]]
        additional_nans = np.full(list(values.shape[:-1]) + offset, np.nan)
        return np.concatenate([additional_nans, smoothed_values], axis=values.ndim - 1)

    def _can_smooth(self, base_values):
        """Returns True if the values can be smoothed, False otherwise.

        If the spec parameters describe valid smoothing and the column dimensions is a
        categorical date it returns True, False otherwise.
        """
        window = self._window
        if base_values.size == 0:
            # --- when base values are empty just return them. Cannot smooth an empty
            # --- array.
            return False
        if not self._dimension_type == DT.CAT_DATE:
            # --- smoothing can be performed only on categorical date dimension type
            orienation = "Row" if base_values.ndim == 1 else "Column"
            warnings.warn(
                "No smoothing performed. {} dimension must be a categorical "
                "date.".format(orienation)
            )
            return False
        # --- window cannot be wider than the number of periods in data
        # --- and it must be at least 2.
        if window > base_values.shape[-1] or window < 2:
            warnings.warn(
                "No smoothing performed. Smoothing window must be between 2 and the "
                "number of periods ({}), got {}".format(base_values.shape[-1], window),
            )
            return False
        return True

    @lazyproperty
    def _window(self):
        """Int representing the window for the moving avarage specified in the dict."""
        return self._smoothing_dict.get("window") or 2
