# LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation

This repository contains the code for the paper Vladan Stojnić, Yannis Kalantidis, Jiří Matas, Giorgos Tolias, ["LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation"](http://arxiv.org/abs/2503.19777), In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

<div align="center">
    
[![arXiv](https://img.shields.io/badge/arXiv-2503.19777-b31b1b.svg)](http://arxiv.org/abs/2503.19777) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/stojnvla/LPOSS)

</div>

## Demo

The demo of our method is available at [<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" height=20px> huggingface spaces](https://huggingface.co/spaces/stojnvla/LPOSS).

## Setup

Setup the conda environment:
```
# Create conda environment
conda create -n lposs python=3.9
conda activate lposs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Install MMCV and MMSegmentation:
```
pip install -U openmim
mim install mmengine    
mim install "mmcv-full==1.6.0"
mim install "mmsegmentation==0.27.0"
```
Install additional requirements:
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install kornia==0.7.4 cupy-cuda11x ftfy omegaconf open_clip_torch==2.26.1 hydra-core wandb
```

## Offline dataset augmentation

To pre-generate an augmented facade dataset (images, masks, and overlays), use the helper script:

```
python tools/augment_dataset.py -c configs/augmentation.yaml
```

Key points:

* The default config (`configs/augmentation.yaml`) expects raw images under `data/facades/images` and masks under `data/facades/masks`, and will write augmented images, masks, and visual overlays into `data/facades_aug/`.
* Images are first split into 448×448 tiles (configurable via `tiling`) so that augmentations are applied on the same patch size the network expects; this avoids random crops that would otherwise change the spatial support post-augmentation.
* When training on the pre-generated tiles, keep the dataloader pipeline free of extra augmentations so you do not stack online transforms on top of the offline ones.
* If mask filenames differ from the image filenames, point `paths.pairs` in the config to a YAML/JSON dict mapping `image_name.png: mask_name.png` so the script can locate the right mask.
* Augmentations include geometric/photometric transforms, weather effects from Albumentations, CutOut, MixUp, and CutMix. Counts/probabilities, output formats, and overlay transparency can all be tuned in the YAML file.
* A tqdm progress bar is shown while augmentations are generated so you can estimate runtime even with multiple augmentations per image.

To split the augmented tiles (e.g., `data/facades_aug/images` and `data/facades_aug/masks`) into train/val/test subsets, use:

```
python tools/split_dataset.py --data-root data/facades_aug --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

By default the script writes the splits to `<data-root>/data_prepared/{train,val,test}/{images,masks}`. Adjust the ratios or the `--output-dir` as needed. Masks must share filenames with their corresponding images.

## Datasets

We use 8 benchmark datasets: PASCAL VOC20, PASCAL Context59, COCO-Object, PASCAL VOC, PASCAL Context, COCO-Stuff, Cityscapes, and ADE20k.

To run the evaluation, download and set up PASCAL VOC, PASCAL Context, COCO-Stuff164k, Cityscapes, and ADE20k datasets following ["MMSegmentation"](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html) data preparation document.

COCO-Object dataset uses only object classes from COCO-Stuff164k dataset by collecting instance segmentation annotations. Run the following command to convert instance segmentation annotations to semantic segmentation annotations:

```
python tools/convert_coco.py data/coco_stuff164k/ -o data/coco_stuff164k/
```

## Running

The provided code can be run using follwing commands:

LPOSS:
```
torchrun main_eval.py lposs.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

LPOSS+:
```
torchrun main_eval.py lposs_plus.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

## Citation

```
@InProceedings{stojnic2025_lposs,
    author    = {Stojni\'c, Vladan and Kalantidis, Yannis and Matas, Ji\v{r}\'i  and Tolias, Giorgos},
    title     = {LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

## Acknowledgments

This repository is based on ["CLIP-DINOiser: Teaching CLIP a few DINO tricks for Open-Vocabulary Semantic Segmentation"](https://github.com/wysoczanska/clip_dinoiser). Thanks to the authors!
