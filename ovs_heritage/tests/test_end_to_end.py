import csv
import json

import numpy as np
from PIL import Image
import torch

from ovs_heritage.losses import combined_two_head_loss
from ovs_heritage.metadata import make_metadata
from ovs_heritage.ontology import load_ontology
from ovs_heritage.projection import OntologyProjection
from ovs_heritage.validate_dataset import validate_splits


def test_cpu_two_map_p0_flow(tmp_path):
    ontology = load_ontology()
    projection = OntologyProjection.canonical_v2()
    main_path = tmp_path / "main.png"
    ornament_path = tmp_path / "ornament.png"
    Image.fromarray(np.array([[7, 5, 11, 255]], dtype=np.uint8)).save(main_path)
    Image.fromarray(np.array([[1, 1, 0, 255]], dtype=np.uint8)).save(ornament_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["main_mask_path", "ornament_mask_path", "facade_id"])
        writer.writeheader()
        writer.writerow({"main_mask_path": main_path.name, "ornament_mask_path": ornament_path.name,
                         "facade_id": "facade_1"})
    report = validate_splits({"test": manifest}, ontology)
    assert report["valid"]

    y_main = torch.tensor([[[7, 5, 11, 255]]])
    y_ornament = torch.tensor([[[1, 1, 0, 255]]])
    main_logits = torch.randn(1, 11, 1, 4, requires_grad=True)
    ornament_logits = torch.randn(1, 1, 1, 4, requires_grad=True)
    channel_target = projection.semantic_main_to_channels(y_main)
    assert channel_target.tolist() == [[[7, 5, 10, 255]]]
    losses = combined_two_head_loss(
        main_logits, ornament_logits, y_main, y_ornament, lambda_ornament=0.5,
    )
    losses.total.backward()
    semantic_prediction = projection.main_logits_to_semantic(main_logits.detach())
    ornament_prediction = projection.ornament_logits_to_binary(
        ornament_logits.detach(), threshold=0.5,
    )
    assert semantic_prediction.shape == y_main.shape
    assert ornament_prediction.shape == ornament_logits.shape
    metadata = make_metadata(
        component_name="p0.synthetic_flow", component_version="1",
        ontology_version=ontology.version, ontology_hash=ontology.hash,
        mapping=projection.as_dict(), validator_schema_version=report["validator_schema_version"],
        source_fingerprints=report["source_fingerprints"], loss_settings=losses.metadata,
        ornament_threshold=0.5,
    )
    assert json.loads(metadata.to_json())["hash"] == metadata.hash
