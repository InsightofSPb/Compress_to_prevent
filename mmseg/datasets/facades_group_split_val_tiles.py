import os

classes = (
    "BACKGROUND",
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
    [0, 0, 0],
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

val_tiles_root = os.environ.get(
    "FACADES_SEG_VAL_TILES_ROOT",
    "data/facades_group_split/segmentation_eval_tiles",
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    val=dict(
        type="CustomDataset",
        data_root=val_tiles_root,
        img_dir="val/images",
        ann_dir="val/masks",
        img_suffix=".png",
        seg_map_suffix=".png",
        ignore_index=255,
        reduce_zero_label=False,
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
