"""The self-occlusion trial changes only evidence-backed unknown points."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.self_occlusion_trial import POLICY, candidates, convert_labels


def evidence():
    shape = (1, 512)
    labels = {
        "schema": np.asarray("mamma_512_label_v1"),
        "policy": np.asarray("old_policy"),
        "person_ids": np.asarray([7], np.int32),
        "visibility_value": np.full(shape, -1, np.int8),
        "visibility_reason_code": np.full(shape, 4, np.uint8),
        "geometry_defined": np.ones(shape, bool),
        "xy_full_erp": np.ones((1, 512, 2), np.float32),
        "physical_domain": np.ones(shape, bool),
        "render_coverage": np.ones(shape, bool),
        "gt_ray_range_m": np.full(shape, 2, np.float32),
        "body_first_range_m": np.full(shape, 1.9, np.float32),
        "front_instance_id": np.full(shape, 7, np.int32),
        "gt_minus_front_range_m": np.full(shape, 0.1, np.float32),
        "nearer_same_front_id_3x3_count": np.full(shape, 5, np.uint8),
        "evidence_neighborhood_complete": np.ones(shape, bool),
    }
    labels["visibility_value"][0, 0] = 1
    labels["visibility_value"][0, 1] = 0
    labels["body_first_range_m"][0, 2] = 1.98
    labels["front_instance_id"][0, 3] = 8
    labels["gt_minus_front_range_m"][0, 4] = 0.04
    labels["nearer_same_front_id_3x3_count"][0, 5] = 4
    labels["physical_domain"][0, 6] = False
    labels["render_coverage"][0, 7] = False
    labels["body_first_range_m"][0, 8] = np.inf
    labels["evidence_neighborhood_complete"][0, 9] = False
    return labels


class SelfOcclusionTrialChecks(unittest.TestCase):
    def test_only_supported_unknown_changes_and_original_stays_intact(self):
        labels = evidence()
        expected = candidates(labels, {0})
        self.assertFalse(expected[0, :10].any())
        self.assertTrue(expected[0, 10:].all())
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "old.npz"
            output = Path(temporary) / "new.npz"
            np.savez_compressed(source, **labels)
            counts = convert_labels(source, output, {0})
            self.assertEqual(counts["changed"], 502)
            with np.load(source, allow_pickle=False) as old, np.load(output, allow_pickle=False) as new:
                self.assertTrue(np.array_equal(old["visibility_value"], labels["visibility_value"]))
                self.assertTrue(np.array_equal(new["visibility_value"][~expected], old["visibility_value"][~expected]))
                self.assertTrue(np.all(new["visibility_value"][expected] == 0))
                self.assertTrue(np.all(new["visibility_reason_code"][expected] == 8))
                self.assertEqual(str(new["policy"]), POLICY)
                for key in old.files:
                    if key not in {"policy", "visibility_value", "visibility_reason_code"}:
                        self.assertTrue(np.array_equal(old[key], new[key]))


if __name__ == "__main__":
    unittest.main()
