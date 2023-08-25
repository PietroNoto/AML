DATA_PATH = '/mnt/c/Users/Matteo/Documents/Blender/hopper/'
PROJECT_PATH = '/mnt/c/Users/Matteo/Documents/Poli/AML 2/vision/'

_base_ = ['./mmpose/configs/_base_/default_runtime.py']

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict( # data normalization and channel transposition
        type='PoseDataPreprocessor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bgr_to_rgb=True
    ),
    backbone=dict( # config of backbone
        type='SCNet',
        depth=50,
        in_channels=3,
        init_cfg=dict(
            type='Pretrained', # load pretrained weights to backbone
            checkpoint='https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50-dc6a7e87.pth'
        )
    ),
    head=dict( # config of head
        type='RegressionHead',
        in_channels=10,
        num_joints=5,
    )
)
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg=dict(
    flip_test=False, # flag of flip test
    flip_mode='heatmap', # heatmap flipping
    shift_heatmap=True,  # shift the flipped heatmap several pixels to get a better performance
)
test_cfg=dict(
    flip_test=False, # flag of flip test
    flip_mode='heatmap', # heatmap flipping
    shift_heatmap=True,  # shift the flipped heatmap several pixels to get a better performance
)

train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_PATH+'train/',
        ann_file=DATA_PATH+'train.json',
        data_prefix=dict(img='train/'),
        # specify the new dataset meta information config file
        metainfo=dict(from_file=PROJECT_PATH+'/hopper_dataset.py'),
    )
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='CocoDataset',
        data_root=DATA_PATH+'test',
        ann_file=DATA_PATH+'test.json',
        data_prefix=dict(img='test/'),
        # specify the new dataset meta information config file
        metainfo=dict(from_file=PROJECT_PATH+'hopper_dataset.py'),
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric', # coco AP
    ann_file=DATA_PATH+'test.json') # path to annotation file
test_evaluator = val_evaluator # use val as test by default

optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005)) # optimizer and initial lr