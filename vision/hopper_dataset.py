dataset_info = dict(
    dataset_name='HopperDataset',
    paper_info=dict(
        author='Carlucci, Francesco and Colucci, Matteo and Noto, Pietro',
        title='Advanced Machine Learning - 2022/2023',
        container='Computer Engineering Master\'s Degree - Politecnico di Torino',
        year='2023',
        homepage='',
    ),
    keypoint_info={
        0: dict(name='torso_top', id=0, color=[0, 255, 0], type='', swap=''),
        1: dict(name='waist', id=1, color=[0, 255, 0], type='', swap=''),
        2: dict(name='knee', id=2, color=[0, 255, 0], type='', swap=''),
        3: dict(name='ankle', id=3, color=[0, 255, 0], type='', swap=''),
        4: dict(name='foot_tip', id=4, color=[0, 255, 0], type='', swap='')
    },
    skeleton_info={
        0: dict(link=('torso_top', 'waist'), id=0, color=[0, 200, 0]),
        1: dict(link=('waist', 'knee'), id=1, color=[0, 200, 0]),
        2: dict(link=('knee', 'ankle'), id=2, color=[0, 200, 0]),
        3: dict(link=('ankle', 'foot_tip'), id=3, color=[0, 200, 0])
    },
    joint_weights=[
        1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.1, 0.1, 0.1, 0.1, 0.1
    ])