# encoding: utf-8

from mock import Mock

from cr.cube.mixins.data_table import DataTable


def test_cube_counts():
    dt = DataTable({'result': {}})
    assert dt.counts == (None, None)

    fake_count = Mock()
    dt = DataTable({'result': {'unfiltered': fake_count}})
    assert dt.counts == (fake_count, None)

    dt = DataTable({'result': {'filtered': fake_count}})
    assert dt.counts == (None, fake_count)

    dt = DataTable({'result': {'unfiltered': fake_count, 'filtered': fake_count}})
    assert dt.counts == (fake_count, fake_count)
