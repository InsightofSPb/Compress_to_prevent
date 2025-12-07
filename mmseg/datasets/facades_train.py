data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CustomDataset",
        data_root="/path/to/data/",
        img_dir="/home/sasha/LPOSS/data_augmented/facades_aug/data_prepared/train/images",
        ann_dir="/home/sasha/LPOSS/data_augmented/facades_aug/data_prepared/train/masks",
        pipeline=[...],
    ),
)