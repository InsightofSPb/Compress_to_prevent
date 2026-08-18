import csv
import json

import numpy as np
from PIL import Image
import pytest

from ovs_heritage.ontology import V1_VERSION, load_ontology, ontology_from_mapping
from ovs_heritage.validate_dataset import (
    V1_DATASET_SCHEMA,
    V2_DATASET_SCHEMA,
    main,
    validate_splits,
)


def validate_v2(sources):
    ontology = load_ontology()
    return validate_splits(
        sources,
        ontology,
        schema_version=V2_DATASET_SCHEMA,
        ontology_version=ontology.version,
    )


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
    report = validate_v2({"test": manifest})
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
        report = validate_v2({"test": manifest})
        assert not report["valid"] and message in " ".join(report["errors"])
    empty = tmp_path / "empty.csv"
    write_v2_manifest(empty, [])
    assert "split is empty" in " ".join(validate_v2({"test": empty})["errors"])


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
    report = validate_v2({"train": manifests[0], "test": manifests[1]})
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
    report = validate_splits(
        {"test": manifest},
        ontology,
        schema_version=V1_DATASET_SCHEMA,
        ontology_version=ontology.version,
    )
    assert not report["valid"] and "invalid legacy-v1 IDs [11]" in " ".join(report["errors"])


def test_cli_writes_report_on_failure(tmp_path):
    manifest = tmp_path / "empty.csv"
    write_v2_manifest(manifest, [])
    output = tmp_path / "report.json"
    ontology = load_ontology()
    assert main([
        "--test", str(manifest),
        "--schema-version", V2_DATASET_SCHEMA,
        "--ontology-version", ontology.version,
        "--output", str(output), "--strict",
    ]) == 1
    assert output.exists() and json.loads(output.read_text())["valid"] is False


def test_schema_and_ontology_declarations_are_required_and_must_match(tmp_path):
    ontology = load_ontology()
    with pytest.raises(TypeError):
        validate_splits({"test": tmp_path / "missing.csv"}, ontology)
    with pytest.raises(ValueError, match="unsupported dataset schema"):
        validate_splits(
            {}, ontology, schema_version="typo", ontology_version=ontology.version,
        )
    with pytest.raises(ValueError, match="does not match loaded"):
        validate_splits(
            {}, ontology, schema_version=V2_DATASET_SCHEMA,
            ontology_version="heritage_facades_v2_typo",
        )
    with pytest.raises(ValueError, match="requires dataset schema"):
        validate_splits(
            {}, ontology, schema_version=V1_DATASET_SCHEMA,
            ontology_version=ontology.version,
        )


def test_conflicting_row_declaration_and_split_statistics(tmp_path):
    good_main = tmp_path / "good_main.png"
    good_ornament = tmp_path / "good_ornament.png"
    save(good_main, [[0]])
    save(good_ornament, [[0]])
    manifest = tmp_path / "mixed.csv"
    fields = [
        "main_mask_path", "ornament_mask_path", "facade_id",
        "schema_version", "ontology_version",
    ]
    ontology = load_ontology()
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "main_mask_path": good_main.name,
            "ornament_mask_path": good_ornament.name,
            "facade_id": "ok",
            "schema_version": V2_DATASET_SCHEMA,
            "ontology_version": ontology.version,
        })
        writer.writerow({
            "main_mask_path": "missing.png",
            "ornament_mask_path": good_ornament.name,
            "facade_id": "bad",
            "schema_version": V2_DATASET_SCHEMA,
            "ontology_version": ontology.version,
        })
    report = validate_v2({"test": manifest})
    split = report["splits"]["test"]
    assert split["manifest_row_count"] == 2
    assert split["valid_sample_count"] == 1
    assert split["failed_sample_count"] == 1
    assert split["split_error_count"] == 0
    assert split["valid_sample_count"] + split["failed_sample_count"] == split["manifest_row_count"]

    conflict = tmp_path / "conflict.csv"
    with conflict.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "main_mask_path": good_main.name,
            "ornament_mask_path": good_ornament.name,
            "facade_id": "f",
            "schema_version": "wrong",
            "ontology_version": ontology.version,
        })
    conflict_report = validate_v2({"test": conflict})
    assert conflict_report["splits"]["test"]["split_error_count"] == 1
    assert conflict_report["splits"]["test"]["failed_sample_count"] == 0


def test_empty_and_unreadable_manifests_are_split_errors(tmp_path):
    empty = tmp_path / "empty_again.csv"
    write_v2_manifest(empty, [])
    empty_split = validate_v2({"test": empty})["splits"]["test"]
    assert empty_split["manifest_row_count"] == 0
    assert empty_split["failed_sample_count"] == 0
    assert empty_split["split_error_count"] == 1

    unreadable = validate_v2({"test": tmp_path / "missing.csv"})["splits"]["test"]
    assert unreadable["manifest_row_count"] == 0
    assert unreadable["failed_sample_count"] == 0
    assert unreadable["split_error_count"] == 1


def test_multiple_invalid_rows_count_as_failed_samples(tmp_path):
    ornament = tmp_path / "ornament.png"
    save(ornament, [[0]])
    manifest = tmp_path / "invalid_rows.csv"
    write_v2_manifest(manifest, [
        {"main_mask_path": "missing_a.png", "ornament_mask_path": ornament.name,
         "facade_id": "a", "source_id": "a"},
        {"main_mask_path": "missing_b.png", "ornament_mask_path": ornament.name,
         "facade_id": "b", "source_id": "b"},
    ])
    split = validate_v2({"test": manifest})["splits"]["test"]
    assert split["manifest_row_count"] == 2
    assert split["valid_sample_count"] == 0
    assert split["failed_sample_count"] == 2
    assert split["split_error_count"] == 0
