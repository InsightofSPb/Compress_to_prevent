import csv
import json
from pathlib import Path

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


def make_v1_ontology():
    with open("ovs_heritage/configs/heritage_vocab.yaml", encoding="utf-8") as stream:
        data = json.load(stream)
    data["version"] = V1_VERSION
    data["classes"] = data["classes"][:11]
    data["classes"][8]["name"] = "ornament_intact"
    data["classes"][8]["aliases"] = []
    data["groups"]["ORNAMENT"] = ["ornament_intact"]
    data["groups"]["HUMAN_ACTIVITY"].remove("advertisements")
    return ontology_from_mapping(data)


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
    ontology = make_v1_ontology()
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
    conflict_split = conflict_report["splits"]["test"]
    assert conflict_split["manifest_row_count"] == 1
    assert conflict_split["split_error_count"] == 0
    assert conflict_split["valid_sample_count"] == 0
    assert conflict_split["failed_sample_count"] == 1
    assert "conflicting schema_version" in conflict_split["sample_errors"][0]


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


def test_two_declaration_errors_count_as_one_failed_row(tmp_path):
    main_mask = tmp_path / "declaration_main.png"
    ornament_mask = tmp_path / "declaration_ornament.png"
    save(main_mask, [[0]])
    save(ornament_mask, [[0]])
    manifest = tmp_path / "two_conflicts.json"
    manifest.write_text(json.dumps([{
        "main_mask_path": main_mask.name,
        "ornament_mask_path": ornament_mask.name,
        "facade_id": "facade",
        "schema_version": "wrong_schema",
        "ontology_version": "wrong_ontology",
    }]))
    split = validate_v2({"test": manifest})["splits"]["test"]
    assert split["manifest_row_count"] == 1
    assert split["valid_sample_count"] == 0
    assert split["failed_sample_count"] == 1
    assert split["split_error_count"] == 0
    assert len(split["sample_errors"]) == 2
    assert "conflicting schema_version" in split["sample_errors"][0]
    assert "conflicting ontology_version" in split["sample_errors"][1]
    assert split["valid_sample_count"] + split["failed_sample_count"] == 1


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


def write_v2_manifest_with_image(path, rows):
    fields = ["image_path", "main_mask_path", "ornament_mask_path", "facade_id", "source_id"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_source_id_and_image_path_leakage_with_different_masks(tmp_path):
    image = tmp_path / "image.png"
    save(image, [[0]])
    manifests = []
    for split in ("train", "test"):
        main_mask = tmp_path / f"{split}_main.png"
        ornament_mask = tmp_path / f"{split}_ornament.png"
        save(main_mask, [[0]])
        save(ornament_mask, [[0]])
        manifest = tmp_path / f"{split}_source.csv"
        write_v2_manifest_with_image(manifest, [{
            "image_path": image.name,
            "main_mask_path": main_mask.name,
            "ornament_mask_path": ornament_mask.name,
            "facade_id": f"facade_{split}",
            "source_id": "repeated_source",
        }])
        manifests.append(manifest)
    report = validate_v2({"train": manifests[0], "test": manifests[1]})
    assert report["source_id_overlaps"][0]["source_ids"] == ["repeated_source"]
    reused = report["duplicated_paths"][0]["paths"]
    assert reused[0]["roles"] == {"train": ["image_path"], "test": ["image_path"]}
    assert report["splits"]["train"]["verified_image_count"] == 1


def test_optional_image_must_be_readable_and_match_mask_grid(tmp_path):
    main_mask = tmp_path / "main.png"
    ornament_mask = tmp_path / "ornament.png"
    save(main_mask, [[0]])
    save(ornament_mask, [[0]])
    for image_name, expected in (("missing.png", "missing or unreadable"), ("corrupt.png", "missing or unreadable")):
        if image_name == "corrupt.png":
            (tmp_path / image_name).write_text("not an image")
        manifest = tmp_path / f"{image_name}.csv"
        write_v2_manifest_with_image(manifest, [{
            "image_path": image_name,
            "main_mask_path": main_mask.name,
            "ornament_mask_path": ornament_mask.name,
            "facade_id": "facade",
            "source_id": image_name,
        }])
        assert expected in " ".join(validate_v2({"test": manifest})["errors"])
    large_image = tmp_path / "large.png"
    save(large_image, [[0, 0]])
    mismatch = tmp_path / "mismatch.csv"
    write_v2_manifest_with_image(mismatch, [{
        "image_path": large_image.name,
        "main_mask_path": main_mask.name,
        "ornament_mask_path": ornament_mask.name,
        "facade_id": "facade",
        "source_id": "large",
    }])
    assert "image/mask grid mismatch" in " ".join(validate_v2({"test": mismatch})["errors"])


def test_facade_leakage_survives_corrupted_mask_and_whitespace_is_rejected(tmp_path):
    ornament = tmp_path / "ornament.png"
    good_main = tmp_path / "good.png"
    save(ornament, [[0]])
    save(good_main, [[0]])
    train = tmp_path / "train_corrupt.csv"
    test = tmp_path / "test_good.csv"
    write_v2_manifest(train, [{
        "main_mask_path": "missing.png", "ornament_mask_path": ornament.name,
        "facade_id": "shared", "source_id": "train",
    }])
    write_v2_manifest(test, [{
        "main_mask_path": good_main.name, "ornament_mask_path": ornament.name,
        "facade_id": "shared", "source_id": "test",
    }])
    report = validate_v2({"train": train, "test": test})
    assert report["facade_overlaps"][0]["facade_ids"] == ["shared"]

    whitespace = tmp_path / "whitespace.csv"
    write_v2_manifest(whitespace, [{
        "main_mask_path": good_main.name, "ornament_mask_path": ornament.name,
        "facade_id": " shared", "source_id": "sample ",
    }])
    errors = " ".join(validate_v2({"test": whitespace})["errors"])
    assert "surrounding whitespace" in errors


def test_v1_relative_directory_content_fingerprint_is_root_independent(tmp_path, monkeypatch):
    ontology = make_v1_ontology()
    first = tmp_path / "first" / "masks"
    second = tmp_path / "second" / "masks"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    save(first / "a.png", [[1]])
    save(second / "a.png", [[1]])
    monkeypatch.chdir(tmp_path)

    def validate_directory(path):
        return validate_splits(
            {"test": path}, ontology,
            schema_version=V1_DATASET_SCHEMA,
            ontology_version=ontology.version,
        )

    first_report = validate_directory(Path("first/masks"))
    second_report = validate_directory(Path("second/masks"))
    assert first_report["valid"] and second_report["valid"]
    assert first_report["source_fingerprints"]["test"] == second_report["source_fingerprints"]["test"]
    save(first / "a.png", [[2]])
    changed = validate_directory(Path("first/masks"))
    assert changed["source_fingerprints"]["test"] != first_report["source_fingerprints"]["test"]


def test_physical_path_leakage_is_role_independent(tmp_path):
    shared = tmp_path / "shared.png"
    train_ornament = tmp_path / "train_ornament.png"
    test_main = tmp_path / "test_main.png"
    save(shared, [[0]])
    save(train_ornament, [[0]])
    save(test_main, [[0]])
    train = tmp_path / "train_roles.csv"
    test = tmp_path / "test_roles.csv"
    write_v2_manifest(train, [{
        "main_mask_path": shared.name,
        "ornament_mask_path": train_ornament.name,
        "facade_id": "train_facade",
        "source_id": "train_source",
    }])
    write_v2_manifest(test, [{
        "main_mask_path": test_main.name,
        "ornament_mask_path": shared.name,
        "facade_id": "test_facade",
        "source_id": "test_source",
    }])
    report = validate_v2({"train": train, "test": test})
    duplicate = report["duplicated_paths"][0]["paths"][0]
    assert duplicate["roles"] == {
        "train": ["main_mask_path"],
        "test": ["ornament_mask_path"],
    }


def test_v2_rejects_one_physical_file_in_multiple_roles(tmp_path):
    shared = tmp_path / "shared_roles.png"
    save(shared, [[0]])
    manifest = tmp_path / "same_row_roles.csv"
    write_v2_manifest(manifest, [{
        "main_mask_path": shared.name,
        "ornament_mask_path": shared.name,
        "facade_id": "facade",
        "source_id": "source",
    }])
    errors = validate_v2({"test": manifest})["splits"]["test"]["sample_errors"]
    assert "one physical file cannot serve multiple roles" in errors[0]


@pytest.mark.parametrize("field,value", [("facade_id", " bad"), ("source_id", 7)])
def test_v1_optional_identifiers_are_strict(tmp_path, field, value):
    ontology = make_v1_ontology()
    mask = tmp_path / "legacy_id.png"
    save(mask, [[0]])
    manifest = tmp_path / f"legacy_{field}.json"
    row = {"mask_path": mask.name, field: value}
    manifest.write_text(json.dumps([row]))
    report = validate_splits(
        {"test": manifest},
        ontology,
        schema_version=V1_DATASET_SCHEMA,
        ontology_version=ontology.version,
    )
    assert report["splits"]["test"]["failed_sample_count"] == 1
    assert field in report["splits"]["test"]["sample_errors"][0]


def test_v1_does_not_coerce_non_string_mask_path(tmp_path):
    ontology = make_v1_ontology()
    manifest = tmp_path / "legacy_numeric_path.json"
    manifest.write_text(json.dumps([{"mask_path": 123}]))
    report = validate_splits(
        {"test": manifest},
        ontology,
        schema_version=V1_DATASET_SCHEMA,
        ontology_version=ontology.version,
    )
    assert "mask_path must be a non-empty string" in " ".join(report["errors"])
