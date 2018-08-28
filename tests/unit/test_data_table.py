# encoding: utf-8

from mock import Mock

from cr.cube.mixins.data_table import DataTable


def test_cube_counts():
    dt = DataTable({})
    assert dt.counts == (None, None)

    fake_count = Mock()
    dt = DataTable({'unfiltered': fake_count})
    assert dt.counts == (fake_count, None)

    dt = DataTable({'filtered': fake_count})
    assert dt.counts == (None, fake_count)

    dt = DataTable({'unfiltered': fake_count, 'filtered': fake_count})
    assert dt.counts == (fake_count, fake_count)
