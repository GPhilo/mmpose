dataset_info = dict(
    dataset_name='lpcorners',
    paper_info=None,
    keypoint_info={
        0:
        dict(name='TL', id=0, color=[79, 193, 232], type='', swap='TR'),
        1:
        dict(
            name='TR',
            id=1,
            color=[160, 213, 104],
            type='',
            swap='TL'),
        2:
        dict(
            name='BR',
            id=2,
            color=[255, 206, 84],
            type='',
            swap='BL'),
        3:
        dict(
            name='BL',
            id=3,
            color=[237, 85, 100],
            type='',
            swap='BR')
    },
    skeleton_info={
        0:
        dict(link=('TL', 'TR'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('TR', 'BR'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('BR', 'BL'), id=2, color=[0, 255, 0]),
        3:
        dict(link=('BL', 'TL'), id=3, color=[0, 255, 0]),
    },
    joint_weights=[
        1., 1., 1., 1.
    ],
    sigmas=[ # These values are basically chosen at random (I took the value for the "eye" class), as I have no clue how to interpret them. They affect evaluation metrics.
        0.025, 0.025, 0.025, 0.025
    ])