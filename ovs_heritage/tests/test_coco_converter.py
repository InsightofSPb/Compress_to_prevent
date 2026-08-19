import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import ovs_heritage.coco_converter as converter
from ovs_heritage.coco_converter import (
    ConversionError,
    ImageResolver,
    convert,
    load_policy,
    polygon_mask,
    preflight,
    validate_manifest,
)


def _fixture(tmp_path: Path, annotations=None, file_name="deadbeef-facade.jpg"):
    images = tmp_path / "source"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(images / "facade.jpg")
    categories = [
        {"id": 91, "name": "DELAMINATION"},
        {"id": 3, "name": "SPALLING"},
        {"id": 44, "name": "ORNAMENT_INTACT"},
    ]
    if annotations is None:
        annotations = [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 91,
                "segmentation": [[1, 1, 7, 1, 7, 7, 1, 7]],
            },
            {
                "id": 2,
                "image_id": 7,
                "category_id": 3,
                "segmentation": [[3, 3, 6, 3, 6, 6, 3, 6]],
            },
            {
                "id": 3,
                "image_id": 7,
                "category_id": 44,
                "segmentation": [
                    [2, 2, 5, 2, 5, 5, 2, 5],
                    [0, 0, 1, 0, 1, 1, 0, 1],
                ],
            },
        ]
    coco = tmp_path / "coco.json"
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 7,
                        "file_name": file_name,
                        "width": 8,
                        "height": 8,
                    }
                ],
                "categories": categories,
                "annotations": annotations,
            }
        )
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "image_id": 7,
                    "facade_id": "F1",
                    "building_id": "B1",
                    "split": "train",
                    "capture_year": 2020,
                }
            ]
        )
    )
    return coco, images, metadata


def test_canonical_coco_polygon_golden_mask():
    actual = polygon_mask([[1, 1, 4, 1, 4, 4, 1, 4]], 6, 6, 10)
    expected = np.zeros((6, 6), dtype=bool)
    expected[1:4, 1:4] = True
    np.testing.assert_array_equal(actual, expected)


def test_canonical_coco_multipart_union():
    actual = polygon_mask(
        [[0, 0, 2, 0, 2, 2, 0, 2], [4, 4, 6, 4, 6, 6, 4, 6]],
        6,
        6,
        11,
    )
    expected = np.zeros((6, 6), dtype=bool)
    expected[0:2, 0:2] = True
    expected[4:6, 4:6] = True
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "segmentation,match",
    [
        ([], "empty segmentation"),
        ([[0, 0, 1, 1]], "three x/y pairs"),
        ([[0, 0, 1, 1, 2, 2]], "degenerate"),
        ([[0, 0, 1, 0, float("nan"), 1]], "finite"),
        ([[20, 20, 21, 20, 21, 21]], "zero decoded area"),
        ({"counts": [], "size": [8, 8]}, "polygon arrays"),
    ],
)
def test_polygon_rejects_malformed_and_degenerate(segmentation, match):
    with pytest.raises(ConversionError, match=match):
        polygon_mask(segmentation, 8, 8, 12)


def test_end_to_end_priority_ornament_and_determinism(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    source_paths = (coco, images / "facade.jpg")
    source_hashes = [sha256(path.read_bytes()).hexdigest() for path in source_paths]
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    convert(coco, images, metadata, first)
    convert(coco, images, metadata, second)
    main = np.asarray(Image.open(first / "main_masks/7.png"))
    ornament = np.asarray(Image.open(first / "ornament_masks/7.png"))
    assert main[3, 3] == 2
    assert ornament[3, 3] == 1 and main[3, 3] == 2
    assert 8 not in np.unique(main)
    assert set(np.unique(main)) <= {0, 2, 3}
    assert set(np.unique(ornament)) <= {0, 1}
    assert ornament[0, 0] == 1
    assert (first / "main_masks/7.png").read_bytes() == (
        second / "main_masks/7.png"
    ).read_bytes()
    assert source_hashes == [
        sha256(path.read_bytes()).hexdigest() for path in source_paths
    ]
    report = json.loads((first / "overlap_report.json").read_text())
    assert any(
        item["overlap_type"]
        == "automatically resolved main + main overlap"
        for item in report["overlaps"]
    )
    assert validate_manifest(first / "manifest.jsonl")["valid"]


def test_annotation_order_does_not_change_masks(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    data = json.loads(coco.read_text())
    data["annotations"].reverse()
    reversed_coco = tmp_path / "reversed.json"
    reversed_coco.write_text(json.dumps(data))
    convert(coco, images, metadata, tmp_path / "a")
    convert(reversed_coco, images, metadata, tmp_path / "b")
    assert (tmp_path / "a/main_masks/7.png").read_bytes() == (
        tmp_path / "b/main_masks/7.png"
    ).read_bytes()


def test_safe_overwrite_rolls_back_and_cleans_staging(tmp_path, monkeypatch):
    coco, images, metadata = _fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original")
    original_write = converter._write_png
    calls = 0

    def fail_late(path, array, allowed):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected late failure")
        original_write(path, array, allowed)

    monkeypatch.setattr(converter, "_write_png", fail_late)
    with pytest.raises(RuntimeError, match="injected late failure"):
        convert(coco, images, metadata, output, overwrite=True)
    assert sentinel.read_text() == "original"
    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]
    assert not list(tmp_path.glob(".output.staging-*"))


def test_audit_preflight_detects_dimensions_and_geometry(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    Image.new("RGB", (9, 8)).save(images / "facade.jpg")
    with pytest.raises(ConversionError, match="dimensions"):
        preflight(coco, images, metadata)
    Image.new("RGB", (8, 8)).save(images / "facade.jpg")
    data = json.loads(coco.read_text())
    data["annotations"][0]["segmentation"] = [[0, 0, 1, 1, 2, 2]]
    coco.write_text(json.dumps(data))
    with pytest.raises(ConversionError, match="degenerate"):
        preflight(coco, images, metadata)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda data: data["categories"].append(
                {"id": 999, "name": "MYSTERY"}
            ),
            "unknown source category",
        ),
        (
            lambda data: data["annotations"].append(
                {
                    "id": 99,
                    "image_id": 999,
                    "category_id": 3,
                    "segmentation": [[0, 0, 1, 0, 1, 1]],
                }
            ),
            "orphan annotation",
        ),
        (
            lambda data: data["categories"].append(
                {"id": 999, "name": "SPALLING"}
            ),
            "source category names must be unique",
        ),
    ],
)
def test_preflight_rejects_invalid_coco(tmp_path, mutation, match):
    coco, images, metadata = _fixture(tmp_path)
    data = json.loads(coco.read_text())
    mutation(data)
    coco.write_text(json.dumps(data))
    with pytest.raises(ConversionError, match=match):
        preflight(coco, images, metadata)


@pytest.mark.parametrize("change,match", [
    (lambda policy: policy["main_priority_high_to_low"].append("crack"), "duplicate"),
    (lambda policy: policy["main_priority_high_to_low"].remove("crack"), "missing"),
    (
        lambda policy: policy["main_priority_high_to_low"].__setitem__(
            -1, "ornament_region"
        ),
        "unknown=.*ornament_region",
    ),
    (
        lambda policy: policy["source_names"].__setitem__("CRACK", "not_real"),
        "unknown ontology targets",
    ),
])
def test_policy_fails_closed(tmp_path, change, match):
    policy = json.loads(converter.POLICY_PATH.read_text())
    change(policy)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ConversionError, match=match):
        load_policy(path)


def test_filename_resolution_rules(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (root / "abcdef12-name.jpg").touch()
    (root / "plain.jpg").touch()
    resolver = ImageResolver(root)
    assert not resolver.resolve("abcdef12-name.jpg")["normalization_applied"]
    assert not resolver.resolve("plain.jpg")["normalization_applied"]
    with pytest.raises(ConversionError, match="missing"):
        resolver.resolve("prefix-plain.jpg")
    (root / "abcdef12-name.jpg").unlink()
    (root / "name.jpg").touch()
    assert ImageResolver(root).resolve("abcdef12-name.jpg")[
        "normalization_applied"
    ]


def test_missing_ambiguous_and_normalization_collision(tmp_path):
    root = tmp_path / "images"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a/x.jpg").touch()
    (root / "b/x.jpg").touch()
    with pytest.raises(ConversionError, match="ambiguous"):
        ImageResolver(root).resolve("x.jpg")
    with pytest.raises(ConversionError, match="missing"):
        ImageResolver(root).resolve("gone.jpg")
    coco, images, metadata = _fixture(tmp_path / "case")
    data = json.loads(coco.read_text())
    data["images"].append(
        {"id": 8, "file_name": "cafebabe-facade.jpg", "width": 8, "height": 8}
    )
    coco.write_text(json.dumps(data))
    with pytest.raises(ConversionError, match="collision"):
        convert(coco, images, metadata, tmp_path / "collision")


def test_manifest_rejects_duplicate_ids_paths_hashes_and_fields(tmp_path):
    coco, images, metadata = _fixture(tmp_path)
    output = tmp_path / "output"
    convert(coco, images, metadata, output)
    manifest = output / "manifest.jsonl"
    row = json.loads(manifest.read_text())
    duplicate = dict(row)
    duplicate["source_coco_sha256"] = "0" * 64
    duplicate["width"] = 0
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(duplicate) + "\n")
    errors = validate_manifest(manifest)["errors"]
    assert any("duplicate sample_id" in error for error in errors)
    assert any("duplicate image_id" in error for error in errors)
    assert any("duplicate artifact path" in error for error in errors)
    assert any("inconsistent source_coco_sha256" in error for error in errors)
    assert any("positive integers" in error for error in errors)


def test_manifest_rejects_missing_facade_and_leakage(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"sample_id": "1"}) + "\n")
    assert not validate_manifest(path)["valid"]
    rows = []
    for index, split in enumerate(("train", "test")):
        rows.append(
            {
                "sample_id": str(index),
                "image_id": index,
                "source_coco_file_name": "x",
                "canonical_file_name": "x",
                "resolved_image_path": "x",
                "image_path": f"missing-{index}",
                "main_mask_path": f"main-{index}",
                "ornament_mask_path": f"ornament-{index}",
                "facade_id": "F",
                "building_id": "B",
                "split": split,
                "schema_version": "heritage_two_map_v2",
                "ontology_version": "heritage_facades_v2_12concepts_two_heads",
                "source_coco_sha256": "0" * 64,
                "source_annotation_ids": [],
                "width": 1,
                "height": 1,
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert any("leaks" in error for error in validate_manifest(path)["errors"])
