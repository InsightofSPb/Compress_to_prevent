import os

masks_dir = "/home/sasha/LPOSS/datasets/SPb_facades/masks"

for fname in os.listdir(masks_dir):
    if "-" in fname:
        new_name = fname.split("-", 1)[1]  # отрезаем всё до первого '-'
        old_path = os.path.join(masks_dir, fname)
        new_path = os.path.join(masks_dir, new_name)
        os.rename(old_path, new_path)