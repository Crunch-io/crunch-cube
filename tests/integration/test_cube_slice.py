from cr.cube.crunch_cube import CrunchCube

from .fixtures import CAT_X_CAT_PRUNING_HS


def test_labels_with_hs_and_pruning():
    cs = CrunchCube(CAT_X_CAT_PRUNING_HS).slices[0]

    # Withouut pruning or H&S
    expected = [
        [
            u'Married', u'Separated', u'Divorced', u'Widowed', u'Single',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Republicans in Congress', u'Both',
            u'Neither', u'Not sure',
        ],
    ]
    actual = cs.labels()
    assert actual == expected

    # Apply pruning
    expected = [
        [
            u'Married', u'Separated', u'Divorced', u'Widowed',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Republicans in Congress', u'Both',
            u'Not sure',
        ],
    ]
    actual = cs.labels(prune=True)
    assert actual == expected

    # Apply H&S
    expected = [
        [
            u'Married', u'left alone', u'Separated', u'Divorced', u'Widowed',
            u'Single', u'Domestic partnership',
        ],
        [
            u'President Obama', u'Obama + Republicans',
            u'Republicans in Congress', u'Both', u'Neither', u'Not sure',
        ],
    ]
    actual = cs.labels(hs_dims=[0, 1])
    assert actual == expected

    # Apply H&S and pruning
    expected = [
        [
            u'Married', u'left alone', u'Separated', u'Divorced', u'Widowed',
            u'Domestic partnership',
        ],
        [
            u'President Obama', u'Obama + Republicans',
            u'Republicans in Congress', u'Both', u'Not sure',
        ],
    ]
    actual = cs.labels(prune=True, hs_dims=[0, 1])
    assert actual == expected
