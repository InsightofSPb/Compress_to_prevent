import os

overlay_dir = "/home/sasha/LPOSS/datasets/SPb_facades/overlays"
masks_dir = "/home/sasha/LPOSS/datasets/SPb_facades/masks"

to_clean = [overlay_dir, masks_dir]
for dir in to_clean:
    for fname in os.listdir(dir):
        if "-" in fname:
            new_name = fname.split("-", 1)[1]  # отрезаем всё до первого '-'
            old_path = os.path.join(dir, fname)
            new_path = os.path.join(dir, new_name)
            os.rename(old_path, new_path)