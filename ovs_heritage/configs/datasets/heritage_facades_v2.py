"""Runtime adapter for MMSeg configs; legacy configs remain untouched."""
from ovs_heritage.ontology import load_ontology
_ONTOLOGY = load_ontology()
ONTOLOGY_VERSION = _ONTOLOGY.version
ONTOLOGY_HASH = _ONTOLOGY.hash
CLASSES = _ONTOLOGY.display_names
PALETTE = _ONTOLOGY.palette
NUM_CLASSES = len(_ONTOLOGY.classes)
EVALUATION_GROUPS = _ONTOLOGY.groups
IGNORE_INDEX = _ONTOLOGY.ignore_index
