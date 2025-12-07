data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(
        type="CustomDataset",
        data_root="data/facades_aug/data_prepared/",
        img_dir="val/images",
        ann_dir="val/masks",
        img_suffix=".png",
        seg_map_suffix=".png",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_semantic_seg"]),
        ],
    ),
)
