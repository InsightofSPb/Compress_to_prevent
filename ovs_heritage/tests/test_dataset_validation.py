import csv
import json

import numpy as np
from PIL import Image

from ovs_heritage.ontology import V1_VERSION, load_ontology, ontology_from_mapping
from ovs_heritage.validate_dataset import main, validate_splits


def save(path, values):
    Image.fromarray(np.asarray(values, dtype=np.uint8)).save(path)


def write_v2_manifest(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["main_mask_path", "ornament_mask_path", "facade_id", "source_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_v2_two_maps_preserve_overlap_advertisements_and_metadata(tmp_path):
    main = tmp_path / "main.png"
    ornament = tmp_path / "ornament.png"
    save(main, [[7, 5, 11, 255]])
    save(ornament, [[1, 1, 0, 255]])
    manifest = tmp_path / "v2.csv"
    write_v2_manifest(manifest, [{"main_mask_path": main.name, "ornament_mask_path": ornament.name,
                                  "facade_id": "facade_1", "source_id": "source_1"}])
    report = validate_splits({"test": manifest}, load_ontology())
    assert report["valid"]
    assert report["splits"]["test"]["valid_sample_count"] == 1
    assert report["splits"]["test"]["main_pixel_count"]["7"] == 1
    assert report["splits"]["test"]["ornament_pixel_count"]["1"] == 2
    assert report["reproducibility"]["hash"]
    assert report["semantic_projection"]["entries"][-1]["semantic_id"] == 8


def test_v2_invalid_labels_shape_missing_facade_and_empty_split(tmp_path):
    cases = [
        ([[8]], [[1]], "f", "invalid Y_main"),
        ([[42]], [[0]], "f", "invalid Y_main"),
        ([[0]], [[2]], "f", "invalid Y_ornament"),
        ([[0, 1]], [[0]], "f", "shape mismatch"),
        ([[0]], [[0]], "", "requires non-empty facade_id"),
    ]
    for index, (main_values, ornament_values, facade, message) in enumerate(cases):
        main = tmp_path / f"m{index}.png"
        ornament = tmp_path / f"o{index}.png"
        save(main, main_values)
        save(ornament, ornament_values)
        manifest = tmp_path / f"case{index}.csv"
        write_v2_manifest(manifest, [{"main_mask_path": main.name, "ornament_mask_path": ornament.name,
                                      "facade_id": facade, "source_id": str(index)}])
        report = validate_splits({"test": manifest}, load_ontology())
        assert not report["valid"] and message in " ".join(report["errors"])
    empty = tmp_path / "empty.csv"
    write_v2_manifest(empty, [])
    assert "split is empty" in " ".join(validate_splits({"test": empty}, load_ontology())["errors"])


def test_facade_and_path_leakage_across_splits(tmp_path):
    main = tmp_path / "main.png"
    ornament = tmp_path / "ornament.png"
    save(main, [[0]])
    save(ornament, [[0]])
    manifests = []
    for split in ("train", "test"):
        manifest = tmp_path / f"{split}.csv"
        write_v2_manifest(manifest, [{"main_mask_path": main.name, "ornament_mask_path": ornament.name,
                                      "facade_id": "same", "source_id": "same"}])
        manifests.append(manifest)
    report = validate_splits({"train": manifests[0], "test": manifests[1]}, load_ontology())
    assert report["facade_overlaps"] and report["duplicated_paths"] and not report["valid"]


def test_v1_explicit_schema_rejects_id11(tmp_path):
    data = json.load(open("ovs_heritage/configs/heritage_vocab.yaml"))
    data["version"] = V1_VERSION
    data["classes"] = data["classes"][:11]
    data["classes"][8]["name"] = "ornament_intact"
    data["classes"][8]["aliases"] = []
    data["groups"]["ORNAMENT"] = ["ornament_intact"]
    data["groups"]["HUMAN_ACTIVITY"].remove("advertisements")
    ontology = ontology_from_mapping(data)
    mask = tmp_path / "legacy.png"
    save(mask, [[11]])
    manifest = tmp_path / "legacy.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["mask_path", "facade_id"])
        writer.writeheader()
        writer.writerow({"mask_path": mask.name, "facade_id": "f"})
    report = validate_splits({"test": manifest}, ontology)
    assert not report["valid"] and "invalid legacy-v1 IDs [11]" in " ".join(report["errors"])


def test_cli_writes_report_on_failure(tmp_path):
    manifest = tmp_path / "empty.csv"
    write_v2_manifest(manifest, [])
    output = tmp_path / "report.json"
    assert main(["--test", str(manifest), "--output", str(output), "--strict"]) == 1
    assert output.exists() and json.loads(output.read_text())["valid"] is False
