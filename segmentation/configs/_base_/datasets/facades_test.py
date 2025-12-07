# Dataset config for facade damage test split
_base_ = None

classes = (
    "CRACK",
    "SPALLING",
    "DELAMINATION",
    "MISSING_ELEMENT",
    "WATER_STAIN",
    "EFFLORESCENCE",
    "CORROSION",
    "ORNAMENT_INTACT",
    "REPAIRS",
    "TEXT_OR_IMAGES",
)

palette = [
    [229, 57, 53],
    [30, 136, 229],
    [67, 160, 71],
    [251, 140, 0],
    [142, 36, 170],
    [253, 216, 53],
    [0, 172, 193],
    [158, 158, 158],
    [78, 158, 158],
    [142, 126, 71],
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    test=dict(
        type="CustomDataset",
        data_root="data/facades_aug/data_prepared/",
        img_dir="test/images",
        ann_dir="test/masks",
        img_suffix=".png",
        seg_map_suffix=".png",
        classes=classes,
        palette=palette,
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations"),
            dict(type="DefaultFormatBundle"),
            dict(
                type="Collect",
                keys=["img", "gt_semantic_seg"],
                meta_keys=("filename", "ori_filename", "ori_shape", "img_shape", "pad_shape"),
            ),
        ],
    ),
)
