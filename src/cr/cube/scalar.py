# encoding: utf-8

"""The cube data-partion object used for 0D means cube."""

from __future__ import absolute_import, division, print_function, unicode_literals

from cr.cube.util import lazyproperty


class MeansScalar(object):
    """Represents slices with means (and no counts)."""

    def __init__(self, means, unweighted_counts):
        self._means = means
        self._unweighted_counts = unweighted_counts

    @lazyproperty
    def means(self):
        return self._means

    @lazyproperty
    def ndim(self):
        """int count of dimensions in this scalar, unconditionally 0.

        A scalar is by definition zero-dimensional.
        """
        return 0

    @lazyproperty
    def table_base(self):
        # TODO: Check why we expect mean instead of the real base in this case.
        return self.means
