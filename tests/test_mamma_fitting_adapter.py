"""Small contract checks for the original-MAMMA 512-vertex fitting adapter."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MammaFittingAdapterChecks(unittest.TestCase):
    def test_mamma_fixture_selects_exact_ordered_vertices(self):
        builder = _module("fitting512_builder", "scripts/fitting512/prepare_worker.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture"
            builder.prepare_fixture(output)
            ids = np.fromfile(output / "mamma_template_vertex_ids.i32", dtype="<i4")
            with np.load(builder.PREDICTION, allow_pickle=False) as prediction:
                self.assertTrue(np.array_equal(ids, prediction["ordered_vertex_ids"]))
            with np.load(builder.MODEL, allow_pickle=False) as smplx:
                selected = np.fromfile(output / "model_cam2_selected_base_v_shaped.f32", dtype="<f4").reshape(512, 3)
                self.assertTrue(np.array_equal(selected, smplx["v_template"][ids]))
            regressor = np.fromfile(output / "densekp_regressor_138x204.f32", dtype="<f4").reshape(512, 512)
            self.assertTrue(np.array_equal(regressor[[0, 255, 511]], np.eye(512, dtype=np.float32)[[0, 255, 511]]))


    def test_rotation_round_trip_on_erp_seam(self):
        adapter = _module("fitting512_compare", "scripts/fitting512/compare.py")
        adapter._external_paths()
        points = np.asarray([[0.001, 1200], [5759.999, 1600], [2800, 700]], np.float32)
        self.assertLess(adapter.verify_rotation(points), 0.002)


    def test_pilot_order_uses_first_matched_frame_in_both_datasets(self):
        adapter = _module("fitting512_compare_order", "scripts/fitting512/compare.py")
        frames = [(('data3', '001'), {}), (('data3', '002'), {}),
                  (('data4', '003'), {}), (('data4', '004'), {})]
        old = {"pairs": [
            {"dataset": d, "source_frame": f,
             "branches": {"original_050": {"person_pairs": [] if f in ('001', '003') else [1]}}}
            for (d, f), _ in frames
        ]}
        ordered = adapter._order_pairs(frames, old)
        self.assertEqual([key for key, _ in ordered[:2]], [('data3', '002'), ('data4', '004')])

    def test_original_input_loader_uses_only_original_prediction(self):
        adapter = _module("fitting512_original_input", "scripts/fitting512/compare.py")
        review = ROOT / "outputs/experiments/real_riding_manual_review_20261024"
        pairs, by_image, prediction = adapter._frozen_original_inputs(review)
        self.assertEqual(len(pairs), 40)
        self.assertEqual(sum(map(len, by_image.values())), 319)
        self.assertEqual(prediction["xy_full_erp"].shape, (319, 512, 2))


if __name__ == "__main__":
    unittest.main()
