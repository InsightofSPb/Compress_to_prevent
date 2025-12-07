data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(
        type="CustomDataset",
        data_root="/path/to/data/",
        img_dir="/home/sasha/LPOSS/data_augmented/facades_aug/data_prepared/val/images",
        ann_dir="/home/sasha/LPOSS/data_augmented/facades_aug/data_prepared/val/masks",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
)
