import json

from ovs_heritage.metadata import make_metadata
from ovs_heritage.ontology import load_ontology
from ovs_heritage.projection import OntologyProjection


def test_metadata_is_deterministic_json_serializable():
    ontology = load_ontology()
    kwargs = dict(
        component_name="test", component_version="1", ontology_version=ontology.version,
        ontology_hash=ontology.hash, mapping=OntologyProjection.canonical_v2().as_dict(),
        validator_schema_version="v2", source_fingerprints={"test": "abc"},
        loss_settings={"lambda_ornament": 0.5, "pos_weight": None}, ornament_threshold=0.5,
    )
    first = make_metadata(**kwargs)
    second = make_metadata(**kwargs)
    assert first.hash == second.hash
    assert json.loads(first.to_json()) == first.to_dict()
