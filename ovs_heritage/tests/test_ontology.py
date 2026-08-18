import json

import numpy as np
import pytest

from ovs_heritage.ontology import (
    OntologyError,
    load_ontology,
    ontology_from_mapping,
    validate_mask_ids,
)

CONFIG = "ovs_heritage/configs/heritage_vocab.yaml"


def config():
    with open(CONFIG, encoding="utf-8") as stream:
        return json.load(stream)


def test_exact_v2_ontology_and_groups():
    ontology = load_ontology()
    assert [item.id for item in ontology.classes] == list(range(12))
    assert ontology.ignore_index == 255 and 255 not in ontology.valid_ids
    assert ontology.by_name("background").id == 0
    assert ontology.by_name("text_or_images").id == 10
    assert ontology.by_name("advertisements").id == 11
    assert ontology.by_name("ornament_region").id == 8
    assert "advertisements" in ontology.groups["HUMAN_ACTIVITY"]
    assert "advertisements" not in ontology.groups["DAMAGE_MACRO"]
    assert tuple(ontology.groups["DAMAGE_MACRO"]) == ontology.class_names[1:8]
    assert len(ontology.palette) == len(set(ontology.palette)) == 12


def test_hash_independent_of_mapping_key_order_and_yaml_format(tmp_path):
    data = config()
    reordered = {key: data[key] for key in reversed(data)}
    yaml_path = tmp_path / "ontology.yaml"
    import yaml
    yaml_path.write_text("# comment\n" + yaml.safe_dump(reordered, sort_keys=False), encoding="utf-8")
    assert load_ontology().hash == load_ontology(yaml_path).hash


@pytest.mark.parametrize("bad_id", [11.0, True, "11"])
def test_ontology_ids_must_be_real_integers(bad_id):
    data = config()
    data["classes"][11]["id"] = bad_id
    with pytest.raises(OntologyError, match=r"classes\[11\]\.id must be an integer"):
        ontology_from_mapping(data)


@pytest.mark.parametrize("version", [
    "heritage_facades_v2_12concepts_two_head",
    "arbitrary_unseen_ontology",
    "",
    2,
])
def test_unknown_empty_and_non_string_versions_are_rejected(version):
    data = config()
    data["version"] = version
    with pytest.raises(OntologyError, match=r"supported versions:.*v1_11classes.*v2_12concepts_two_heads"):
        ontology_from_mapping(data)


def test_unknown_version_is_rejected_before_other_corruption():
    data = config()
    data["version"] = "future_unregistered_version"
    data["ignore_index"] = 254
    data["classes"][9], data["classes"][11] = data["classes"][11], data["classes"][9]
    with pytest.raises(OntologyError, match=r"version 'future_unregistered_version' is unsupported"):
        ontology_from_mapping(data)


@pytest.mark.parametrize(
    ("class_index", "field", "value", "error_path"),
    [
        (11, "is_heritage", "false", r"classes\[11\]\.is_heritage"),
        (3, "prompts", "abc", r"classes\[3\]\.prompts"),
        (11, "aliases", "ad", r"classes\[11\]\.aliases"),
        (11, "evaluation_groups", "HUMAN_ACTIVITY", r"classes\[11\]\.evaluation_groups"),
        (2, "name", 123, r"classes\[2\]\.name"),
        (4, "id", True, r"classes\[4\]\.id"),
        (5, "color", [0, True, 2], r"classes\[5\]\.color\[1\]"),
    ],
)
def test_class_schema_rejects_coercible_wrong_types(class_index, field, value, error_path):
    data = config()
    data["classes"][class_index][field] = value
    with pytest.raises(OntologyError, match=error_path):
        ontology_from_mapping(data)


def test_prompt_and_alias_lists_accept_only_schema_valid_lists():
    data = config()
    data["classes"][0]["prompts"] = ["a valid non-empty prompt"]
    data["classes"][0]["aliases"] = []
    ontology = ontology_from_mapping(data)
    assert ontology.classes[0].prompts == ("a valid non-empty prompt",)
    assert ontology.classes[0].aliases == ()


def test_top_level_group_schema_paths_are_strict():
    data = config()
    data["groups"]["HUMAN_ACTIVITY"] = "advertisements"
    with pytest.raises(OntologyError, match=r"evaluation_groups\.HUMAN_ACTIVITY"):
        ontology_from_mapping(data)


def test_duplicate_ids_names_and_aliases_are_rejected():
    mutations = (
        lambda data: data["classes"][1].__setitem__("id", 0),
        lambda data: data["classes"][1].__setitem__("name", "background"),
        lambda data: data["classes"][1].__setitem__("aliases", ["rust"]),
    )
    for mutate in mutations:
        data = config()
        mutate(data)
        with pytest.raises(OntologyError):
            ontology_from_mapping(data)


def test_strict_versions_and_canonical_order():
    data = config()
    data["ignore_index"] = 254
    with pytest.raises(OntologyError, match="ignore_index=255"):
        ontology_from_mapping(data)

    data = config()
    data["classes"][9]["name"], data["classes"][11]["name"] = (
        data["classes"][11]["name"], data["classes"][9]["name"]
    )
    with pytest.raises(OntologyError, match="canonical class order"):
        ontology_from_mapping(data)

    data = config()
    data["classes"][11]["id"] = 12
    with pytest.raises(OntologyError, match="ordered IDs"):
        ontology_from_mapping(data)


def test_v1_is_exactly_zero_through_ten():
    data = config()
    data["version"] = "heritage_facades_v1_11classes"
    data["classes"] = data["classes"][:11]
    data["groups"]["HUMAN_ACTIVITY"].remove("advertisements")
    v1 = ontology_from_mapping(data)
    assert v1.class_names[-1] == "text_or_images"
    assert v1.valid_ids == frozenset(range(11))


def test_groups_are_bidirectionally_consistent():
    data = config()
    data["classes"][11]["evaluation_groups"] = []
    with pytest.raises(OntologyError, match="top-level group HUMAN_ACTIVITY contains advertisements"):
        ontology_from_mapping(data)

    data = config()
    data["groups"]["HUMAN_ACTIVITY"].remove("advertisements")
    with pytest.raises(OntologyError, match="advertisements in HUMAN_ACTIVITY"):
        ontology_from_mapping(data)


def test_real_non_json_yaml_and_malformed_yaml(tmp_path):
    yaml_path = tmp_path / "plain.yaml"
    yaml_path.write_text("""
version: heritage_facades_v1_11classes
ignore_index: 255
groups: {}
classes: []
""", encoding="utf-8")
    with pytest.raises(OntologyError, match="non-empty list"):
        load_ontology(yaml_path)  # parsed as YAML, then semantically rejected

    malformed = tmp_path / "bad.yaml"
    malformed.write_text("version: [unterminated", encoding="utf-8")
    with pytest.raises(OntologyError, match="malformed YAML"):
        load_ontology(malformed)


def test_mask_dtype_is_checked_before_values_are_converted():
    ontology = load_ontology()
    assert validate_mask_ids(np.array([11, 255], dtype=np.uint8), ontology) == {11, 255}
    for array in (
        np.array([11.5, 255.9]),
        np.array([11.0, 255.0]),
        np.array([True, False]),
        np.array(["11", "255"]),
        np.array([11], dtype=object),
    ):
        with pytest.raises(OntologyError, match=r"dtype.*found IDs"):
            validate_mask_ids(array, ontology, "typed-mask.npy")


def test_unknown_ids_are_explicit():
    with pytest.raises(OntologyError, match=r"mock.png: unknown mask IDs \[17\]"):
        validate_mask_ids(np.array([17], dtype=np.int16), load_ontology(), "mock.png")


def test_ornament_region_is_canonical_and_legacy_alias_is_explicit():
    ontology = load_ontology()
    assert ontology.by_name("ornament_region").id == 8
    with pytest.raises(OntologyError, match="unknown canonical"):
        ontology.by_name("ornament_intact")
    with pytest.raises(OntologyError, match="explicit resolution"):
        ontology.resolve_name("ornament_intact")
    assert ontology.resolve_name("ornament_intact", allow_deprecated_alias=True).name == "ornament_region"
    with pytest.raises(OntologyError, match="unknown canonical"):
        ontology.by_name("does_not_exist")
