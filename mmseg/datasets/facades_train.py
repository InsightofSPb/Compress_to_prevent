data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="CustomDataset",
        data_root="data/facades_aug/data_prepared/",
        img_dir="train/images",
        ann_dir="train/masks",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
)
