from pathlib import Path

import numpy as np
from PIL import Image

from lmms_eval.tasks.roboafford import utils


def test_roboafford_parses_official_coordinate_formats():
    points = utils._parse_points("[(0.25, 0.5)] and (0.75, 0.25)")

    assert points == [(0.25, 0.5), (0.75, 0.25)]


def test_roboafford_scores_normalized_points_inside_mask(tmp_path: Path):
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 2] = 255
    Image.fromarray(mask).save(mask_path)

    assert utils._point_accuracy([(0.2, 0.5)], str(mask_path)) == 1.0
    assert utils._point_accuracy([(0.9, 0.9)], str(mask_path)) == 0.0
    assert utils._point_accuracy([], str(mask_path)) == 0.0


def test_roboafford_aggregates_categories():
    results = [
        {"accuracy": 1.0, "category": "object affordance"},
        {"accuracy": 0.0, "category": "object affordance"},
        {"accuracy": 1.0, "category": "spatial affordance"},
    ]

    assert utils.roboafford_aggregate_overall(results) == 2 / 3
    assert utils.roboafford_aggregate_object_affordance(results) == 0.5
    assert utils.roboafford_aggregate_object_reference(results) == 0.0
    assert utils.roboafford_aggregate_spatial_affordance(results) == 1.0
