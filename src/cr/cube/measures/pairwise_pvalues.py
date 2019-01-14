# encoding: utf-8

"""P-values of pairwise comparison or columns of a contingency table
"""

from cr.cube.distributions.wishart import wishartCDF
from cr.cube.util import lazyproperty


class PairwisePvalues(object):
    """Value object providing matrix of pairwise-comparison P-values"""

    def __init__(self, chisq, shape, axis=0):
        if axis != 0:
            raise NotImplementedError("Pairwise p-values only implemented for colums")
        self.axis = axis
        self._chisq = chisq
        self.n_max = min(shape) - 1
        self.n_min = max(shape) - 1

    @lazyproperty
    def values(self):
        return 1.0 - wishartCDF(self._chisq, self.n_max, self.n_min)
