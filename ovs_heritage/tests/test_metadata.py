import json

from ovs_heritage.metadata import make_metadata
from ovs_heritage.ontology import load_ontology
from ovs_heritage.projection import OntologyProjection


def test_metadata_is_deterministic_json_serializable():
    ontology = load_ontology()
    kwargs = dict(
        component_name="test", component_version="1", ontology_version=ontology.version,
        ontology_hash=ontology.hash, mapping=OntologyProjection.from_ontology(load_ontology()).as_dict(),
        validator_schema_version="v2", source_fingerprints={"test": "abc"},
        loss_settings={"lambda_ornament": 0.5, "pos_weight": None}, ornament_threshold=0.5,
    )
    first = make_metadata(**kwargs)
    second = make_metadata(**kwargs)
    assert first.hash == second.hash
    assert json.loads(first.to_json()) == first.to_dict()


def test_nested_caller_mutation_cannot_change_record_and_exports_are_copies():
    nested = {"entries": [{"semantic_id": 1}]}
    record = make_metadata(
        component_name="test", component_version="1", ontology_version="v",
        ontology_hash="h", mapping=nested,
    )
    original_hash = record.hash
    nested["entries"][0]["semantic_id"] = 99
    exported = record.to_dict()
    exported["payload"]["mapping"]["entries"][0]["semantic_id"] = 42
    assert record.hash == original_hash
    assert record.to_dict()["payload"]["mapping"]["entries"][0]["semantic_id"] == 1


def test_metadata_rejects_invalid_values_and_thresholds():
    import pytest

    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError):
            make_metadata(
                component_name="test", component_version="1", ontology_version="v",
                ontology_hash="h", mapping={}, ornament_threshold=value,
            )
    with pytest.raises(ValueError, match="non-empty"):
        make_metadata(
            component_name="", component_version="1", ontology_version="v",
            ontology_hash="h", mapping={},
        )
    with pytest.raises(TypeError, match="not JSON serializable"):
        make_metadata(
            component_name="test", component_version="1", ontology_version="v",
            ontology_hash="h", mapping={"bad": object()},
        )
