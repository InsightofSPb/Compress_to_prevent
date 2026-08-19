import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ovs_heritage.coco_converter import ConversionError, ImageResolver, convert, inspect_coco, validate_manifest


def _fixture(tmp_path: Path, annotations=None, file_name="deadbeef-facade.jpg"):
    images = tmp_path / "source"; images.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(images / "facade.jpg")
    categories = [{"id": 91, "name": "DELAMINATION"}, {"id": 3, "name": "SPALLING"},
                  {"id": 44, "name": "ORNAMENT_INTACT"}]
    annotations = annotations or [
        {"id": 1, "image_id": 7, "category_id": 91, "segmentation": [[1, 1, 7, 1, 7, 7, 1, 7]]},
        {"id": 2, "image_id": 7, "category_id": 3, "segmentation": [[3, 3, 6, 3, 6, 6, 3, 6]]},
        {"id": 3, "image_id": 7, "category_id": 44, "segmentation": [[2, 2, 5, 2, 5, 5, 2, 5], [0, 0, 1, 0, 1, 1, 0, 1]]},
    ]
    coco = tmp_path / "coco.json"
    coco.write_text(json.dumps({"images": [{"id": 7, "file_name": file_name, "width": 8, "height": 8}],
                                "categories": categories, "annotations": annotations}))
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{"image_id": 7, "facade_id": "F1", "building_id": "B1", "split": "train", "capture_year": 2020}]))
    return coco, images, metadata


def test_end_to_end_mapping_priority_ornament_and_determinism(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    source_hashes = [sha256(p.read_bytes()).hexdigest() for p in (coco, images / "facade.jpg")]
    first, second = tmp_path / "out1", tmp_path / "out2"
    convert(coco, images, metadata, first); convert(coco, images, metadata, second)
    main = np.asarray(Image.open(first / "main_masks/7.png")); ornament = np.asarray(Image.open(first / "ornament_masks/7.png"))
    assert main[3, 3] == 2  # mapping follows the name, not category IDs; spalling wins
    assert ornament[3, 3] == 1 and main[3, 3] == 2
    assert 8 not in np.unique(main)
    assert set(np.unique(main)) <= {0, 2, 3} and set(np.unique(ornament)) <= {0, 1}
    assert ornament[0, 0] == 1  # second polygon in one annotation
    assert (first / "main_masks/7.png").read_bytes() == (second / "main_masks/7.png").read_bytes()
    assert source_hashes == [sha256(p.read_bytes()).hexdigest() for p in (coco, images / "facade.jpg")]
    report = json.loads((first / "overlap_report.json").read_text())
    assert any(item["overlap_type"] == "automatically resolved main + main overlap" for item in report["overlaps"])
    assert validate_manifest(first / "manifest.jsonl")["valid"]


def test_annotation_order_does_not_change_masks(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    data = json.loads(coco.read_text()); data["annotations"].reverse(); reversed_coco = tmp_path / "reversed.json"; reversed_coco.write_text(json.dumps(data))
    convert(coco, images, metadata, tmp_path / "a"); convert(reversed_coco, images, metadata, tmp_path / "b")
    assert (tmp_path / "a/main_masks/7.png").read_bytes() == (tmp_path / "b/main_masks/7.png").read_bytes()


@pytest.mark.parametrize("mutation,match", [
    (lambda data: data["categories"].append({"id": 999, "name": "MYSTERY"}), "unknown source category"),
    (lambda data: data["annotations"].append({"id": 99, "image_id": 999, "category_id": 3, "segmentation": [[0,0,1,0,1,1]]}), "orphan annotation"),
    (lambda data: data["annotations"][0].update(segmentation={"counts": [], "size": [8,8]}), "polygons required"),
    (lambda data: data["annotations"][0].update(segmentation=[[0, 0, float("nan"), 0, 1, 1]]), "finite"),
])
def test_invalid_coco_fails(tmp_path, mutation, match):
    coco, _, _ = _fixture(tmp_path); data = json.loads(coco.read_text()); mutation(data); coco.write_text(json.dumps(data))
    with pytest.raises(ConversionError, match=match): inspect_coco(coco)


def test_filename_resolution_rules(tmp_path):
    root = tmp_path / "images"; root.mkdir(); (root / "abcdef12-name.jpg").touch(); (root / "plain.jpg").touch()
    resolver = ImageResolver(root)
    assert not resolver.resolve("abcdef12-name.jpg")["normalization_applied"]
    assert not resolver.resolve("plain.jpg")["normalization_applied"]
    with pytest.raises(ConversionError, match="missing"): resolver.resolve("prefix-plain.jpg")
    (root / "abcdef12-name.jpg").unlink(); (root / "name.jpg").touch()
    assert ImageResolver(root).resolve("abcdef12-name.jpg")["normalization_applied"]


def test_missing_ambiguous_and_normalization_collision(tmp_path):
    root = tmp_path / "images"; (root / "a").mkdir(parents=True); (root / "b").mkdir();
    (root / "a/x.jpg").touch(); (root / "b/x.jpg").touch()
    with pytest.raises(ConversionError, match="ambiguous"): ImageResolver(root).resolve("x.jpg")
    with pytest.raises(ConversionError, match="missing"): ImageResolver(root).resolve("gone.jpg")
    coco, images, metadata = _fixture(tmp_path / "case")
    data = json.loads(coco.read_text()); data["images"].append({"id": 8, "file_name": "cafebabe-facade.jpg", "width": 8, "height": 8}); coco.write_text(json.dumps(data))
    with pytest.raises(ConversionError, match="collision"): convert(coco, images, metadata, tmp_path / "collision")


def test_size_metadata_and_manifest_failures(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    Image.new("RGB", (9, 8)).save(images / "facade.jpg")
    with pytest.raises(ConversionError, match="dimensions"): convert(coco, images, metadata, tmp_path / "bad-size")
    metadata.write_text(json.dumps([{"image_id": 7, "split": "train"}]))
    with pytest.raises(ConversionError, match="facade_id"): convert(coco, images, metadata, tmp_path / "bad-meta")


def test_manifest_rejects_missing_facade_and_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"sample_id": "1"}) + "\n")
    assert not validate_manifest(path)["valid"]
    rows = [{"sample_id": str(i), "image_id": i, "source_coco_file_name": "x", "canonical_file_name": "x",
             "resolved_image_path": "x", "image_path": "missing", "main_mask_path": "missing", "ornament_mask_path": "missing",
             "facade_id": "F", "building_id": "B", "split": split, "schema_version": "heritage_two_map_v2",
             "ontology_version": "heritage_facades_v2_12concepts_two_heads", "source_coco_sha256": "x",
             "source_annotation_ids": [], "width": 1, "height": 1} for i, split in enumerate(("train", "test"))]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert any("leaks" in error for error in validate_manifest(path)["errors"])
