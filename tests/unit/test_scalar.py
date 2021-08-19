# encoding: utf-8

"""Unit test suite for the  cr.cube.measures.scale_means module."""

from cr.cube.scalar import MeansScalar


class Describe_MeansScalar:
    def it_knows_its_ndim(self):
        assert MeansScalar(None, None).ndim == 0
