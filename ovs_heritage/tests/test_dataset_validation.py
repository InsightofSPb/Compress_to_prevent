import csv
import json

import numpy as np
from PIL import Image

from ovs_heritage.ontology import load_ontology, ontology_from_mapping
from ovs_heritage.validate_dataset import main, validate_splits


def save_png(path, values):
    Image.fromarray(np.array(values, dtype=np.uint8)).save(path)


def test_report_unknown_filename_and_json_is_preserved(tmp_path):
    good = tmp_path / "good.png"
    bad = tmp_path / "unknown.png"
    save_png(good, [[0, 1, 10, 11, 255]])
    save_png(bad, [[42]])
    report = validate_splits({"train": tmp_path}, load_ontology())
    assert not report["valid"]
    assert report["splits"]["train"]["unknown_ids"] == [42]
    assert report["splits"]["train"]["files_with_unknown_ids"] == [
        {"file": str(bad), "ids": [42]}
    ]
    assert "42" in " ".join(report["errors"])

    output = tmp_path / "report.json"
    assert main(["--train", str(tmp_path), "--output", str(output), "--strict"]) == 1
    saved = json.loads(output.read_text())
    assert saved["splits"]["train"]["unknown_ids"] == [42]
    assert str(bad) in json.dumps(saved)


def test_npy_float_and_boolean_masks_are_rejected_with_dtype_values_and_filename(tmp_path):
    masks = {
        "fractional.npy": np.array([[11.5, 255.9]]),
        "integral_float.npy": np.array([[11.0, 255.0]]),
        "boolean.npy": np.array([[True, False]]),
    }
    for name, array in masks.items():
        np.save(tmp_path / name, array)
    report = validate_splits({"test": tmp_path}, load_ontology())
    assert not report["valid"]
    errors = "\n".join(report["errors"])
    for name in masks:
        assert name in errors
    assert "float64" in errors and "bool" in errors
    assert "11.5" in errors and "11.0" in errors


def test_facade_overlap_and_absent_advertisements_warning(tmp_path):
    mask = tmp_path / "no_ads.png"
    save_png(mask, [[0, 1, 255]])
    manifests = []
    for split in ("train", "test"):
        path = tmp_path / f"{split}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["mask_path", "facade_id"])
            writer.writeheader()
            writer.writerow({"mask_path": "no_ads.png", "facade_id": "same"})
        manifests.append(path)
    report = validate_splits({"train": manifests[0], "test": manifests[1]}, load_ontology())
    assert any("overlap" in error for error in report["errors"])
    assert any("ADVERTISEMENTS" in warning for warning in report["warnings"])


def test_id_11_is_valid_in_v2_and_rejected_in_v1(tmp_path):
    mask = tmp_path / "advertisement.png"
    save_png(mask, [[11]])
    assert validate_splits({"test": tmp_path}, load_ontology())["valid"]

    with open("ovs_heritage/configs/heritage_vocab.yaml", encoding="utf-8") as stream:
        data = json.load(stream)
    data["version"] = "heritage_facades_v1_11classes"
    data["classes"] = data["classes"][:11]
    data["groups"]["HUMAN_ACTIVITY"].remove("advertisements")
    report = validate_splits({"test": tmp_path}, ontology_from_mapping(data))
    assert not report["valid"]
    assert report["splits"]["test"]["unknown_ids"] == [11]
    assert "advertisement.png" in "\n".join(report["errors"])


def test_tile_manifest_counts_unique_non_empty_source_ids(tmp_path):
    mask_paths = []
    for name in ("tile_a.png", "tile_b.png", "tile_c.png"):
        path = tmp_path / name
        save_png(path, [[0, 11]])
        mask_paths.append(path)
    manifest = tmp_path / "tiles.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "mask_path", "facade_id"])
        writer.writeheader()
        writer.writerows([
            {"source_id": "facade_001", "mask_path": mask_paths[0].name, "facade_id": "f1"},
            {"source_id": "facade_001", "mask_path": mask_paths[1].name, "facade_id": "f1"},
            {"source_id": "facade_002", "mask_path": mask_paths[2].name, "facade_id": "f2"},
        ])
    split = validate_splits({"test": manifest}, load_ontology())["splits"]["test"]
    assert split["image_count"] == 2
    assert split["mask_count"] == 3
    assert split["tile_count"] == 3
    assert split["image_count_source"] == "unique non-empty source_id"


def test_image_manifest_counts_rows_and_empty_tile_source_id_is_error(tmp_path):
    for name in ("image_a.png", "image_b.png"):
        save_png(tmp_path / name, [[0, 11]])
    ordinary = tmp_path / "ordinary.csv"
    with ordinary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mask_path", "facade_id"])
        writer.writeheader()
        writer.writerows([
            {"mask_path": "image_a.png", "facade_id": "f1"},
            {"mask_path": "image_b.png", "facade_id": "f2"},
        ])
    split = validate_splits({"test": ordinary}, load_ontology())["splits"]["test"]
    assert split["image_count"] == 2 and split["mask_count"] == 2
    assert split["tile_count"] is None

    invalid = tmp_path / "invalid_tiles.csv"
    with invalid.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "mask_path"])
        writer.writeheader()
        writer.writerow({"source_id": "", "mask_path": "image_a.png"})
    report = validate_splits({"test": invalid}, load_ontology())
    assert not report["valid"]
    assert "empty source_id" in "\n".join(report["errors"])
