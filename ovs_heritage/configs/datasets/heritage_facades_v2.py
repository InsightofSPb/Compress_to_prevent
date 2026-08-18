"""Metadata adapter for the v2 two-target representation; not a model config."""

from ovs_heritage.ontology import load_ontology
from ovs_heritage.projection import OntologyProjection

_ONTOLOGY = load_ontology()
_PROJECTION = OntologyProjection.from_ontology(_ONTOLOGY)
ONTOLOGY_VERSION = _ONTOLOGY.version
ONTOLOGY_HASH = _ONTOLOGY.hash
DATASET_SCHEMA_VERSION = "heritage_two_map_v2"
SEMANTIC_CONCEPTS = _ONTOLOGY.display_names
PALETTE = _ONTOLOGY.palette
MAIN_SEMANTIC_IDS = tuple(entry.semantic_id for entry in _PROJECTION.main_entries)
MAIN_CLASSES = tuple(entry.canonical_name for entry in _PROJECTION.main_entries)
MAIN_NUM_CHANNELS = _PROJECTION.main_channel_count
ORNAMENT_SEMANTIC_ID = 8
ORNAMENT_NUM_CHANNELS = 1
OUTPUT_MAPPING = _PROJECTION.as_dict()
EVALUATION_GROUPS = _ONTOLOGY.groups
IGNORE_INDEX = _ONTOLOGY.ignore_index
