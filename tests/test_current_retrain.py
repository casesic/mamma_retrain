"""CPU checks for the one current MAMMA data, task and resume contract."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from mamma_ext.data import MammaSavedGTDataset
from mamma_ext.policy import (
    active_task_loss,
    configure_active_optimizer,
    load_active_policy,
)
from mamma_ext.model import restore_xy
from mamma_ext.recovery import (
    capture_rng,
    load_evaluation_boundary,
    load_regular_boundary,
    save_regular_boundary,
)
from mamma_ext.runner import _execution_preflight
from mamma_ext.split import validate_disjoint
from mamma_ext.upstream import crop_math

ROOT = Path(__file__).resolve().parents[1]
REAL_MANIFEST = (
    ROOT.parent
    / "human_data_generator/outputs/runs/first_current_render_a1_20261023_pair_walk_densekp_env/observations.json"
)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def tiny_bundle(root):
    """A one-target metadata fixture. The PNGs are never decoded in these tests."""
    root = Path(root)
    (root / "roi.png").write_bytes(b"fixture")
    (root / "mask.png").write_bytes(b"fixture")
    label = dict(
        schema=np.array("mamma_512_label_v1"),
        person_ids=np.array([7]),
        frame_index=np.array(3),
        camera_id=np.array("cam2"),
        rig_id=np.array("rig_side"),
        scene_episode_id=np.array("fixture"),
        time_tick=np.array(3),
        time_base=np.array([1, 8]),
        asset_frame_index=np.array([9]),
        source_frame_index=np.array([11]),
        template_id=np.array("MAMMA_verts_512_ordered_v1"),
        ordered_vertex_ids=np.arange(512),
        render_coverage_xyxy=np.array([0, 0, 64, 64]),
        full_image_size_wh=np.array([64, 64]),
        xy_full_erp=np.zeros((1, 512, 2)),
        visibility_value=np.full((1, 512), -1),
    )
    annotation = dict(
        person_ids=np.array([7]),
        frame_index=np.array(3),
        scene_episode_id=np.array("fixture"),
        rig_id=np.array("rig_side"),
        time_tick=np.array(3),
        time_base=np.array([1, 8]),
        person_asset_frame=np.array([9]),
        person_source_frame=np.array([11]),
        camera_to_world_erp=np.eye(4),
    )
    np.savez(root / "labels.npz", **label)
    np.savez(root / "annotations.npz", **annotation)
    row = dict(
        episode="fixture",
        rig="rig_side",
        frame_index=3,
        camera_id="cam2",
        person_id=7,
        person_index=0,
        status="observed",
        absent_reason=None,
        rgb="roi.png",
        mask="mask.png",
        labels="labels.npz",
        annotations="annotations.npz",
        time_tick=3,
        time_base=[1, 8],
        asset_frame_index=9,
        source_frame_index=11,
        roi_xyxy=[0, 0, 64, 64],
        full_image_size_wh=[64, 64],
        crop_support={"crop_policy_version": "mask_bbox_inclusive_v1"},
    )
    bundle = dict(
        schema="mamma_bundle_v1",
        episode="fixture",
        frame_count=8,
        frames=[3],
        rig_ids=["rig_side"],
        camera_ids=["cam2"],
        template_id="MAMMA_verts_512_ordered_v1",
        ordered_vertex_ids=list(range(512)),
        observations=[row],
    )
    write_json(root / "observations.json", bundle)
    write_json(root / "reader.json", {"schema": "mamma_reader_v1", "manifest": "observations.json"})
    return bundle, label, annotation


class CurrentRetrainTest(unittest.TestCase):
    def test_bundle_identity_time_template_and_files_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, label, annotation = tiny_bundle(root)
            reader = root / "reader.json"
            self.assertEqual(len(MammaSavedGTDataset(reader)), 1)

            for changed, error in (
                ({"frame_index": 4}, "bundle_identity_or_status_conflict"),
                ({"roi_xyxy": [1, 0, 64, 64]}, "saved_gt_identity_time_or_template_conflict"),
                ({"labels": "missing.npz"}, "missing_or_external_bundle_file:labels"),
            ):
                modified = json.loads(json.dumps(bundle))
                modified["observations"][0].update(changed)
                write_json(root / "observations.json", modified)
                with self.assertRaisesRegex(ValueError, error):
                    MammaSavedGTDataset(reader)
            write_json(root / "observations.json", bundle)
            for key, changed in (
                ("time_tick", np.array(4)),
                ("ordered_vertex_ids", np.arange(512)[::-1]),
            ):
                altered = dict(label, **{key: changed})
                np.savez(root / "labels.npz", **altered)
                with self.assertRaisesRegex(
                    ValueError, "saved_gt_identity_time_or_template_conflict"
                ):
                    MammaSavedGTDataset(reader)
            np.savez(root / "labels.npz", **label)
            np.savez(root / "annotations.npz", **annotation)

    def test_current_policy_relative_paths_and_old_schema_reject(self):
        config = ROOT / "configs/retrain/mamma_regular.json"
        current, dataset = load_active_policy(config, manifest_override=REAL_MANIFEST)
        self.assertEqual(current["schema"], "mamma_training_v2")
        self.assertEqual(current["input_config"], str(config.parent / "mamma_saved_gt_reader.json"))
        self.assertEqual(len(dataset), 4)
        self.assertFalse(current["execution_authorized"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.json"
            old = json.loads(config.read_text())
            old["schema"] = "mamma_training_v1"
            old["supervised_visibility"] = True
            write_json(path, old)
            with self.assertRaisesRegex(ValueError, "unsupported_active_training_policy"):
                load_active_policy(path, manifest_override=REAL_MANIFEST)

            for task in ("coordinate", "visibility"):
                for invalid in (0, 1, "true", None):
                    malformed = json.loads(config.read_text())
                    malformed["loss_enabled"][task] = invalid
                    write_json(path, malformed)
                    with self.assertRaisesRegex(ValueError, "task_loss_optimizer_mismatch"):
                        load_active_policy(path, manifest_override=REAL_MANIFEST)

    def test_loss_unknown_empty_and_optimizer_membership(self):
        prediction = dict(
            joints2d=torch.tensor([[[0.2, 0.1, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, -0.1]]]),
            visibility=torch.tensor([[[0.5], [-0.3], [2.0]]]),
        )
        batch = dict(
            xy_target=torch.zeros((1, 3, 2)),
            coordinate_loss_valid=torch.tensor([[True, True, False]]),
            visibility=torch.tensor([[1, 0, -1]]),
            visibility_loss_valid=torch.tensor([[True, True, False]]),
        )
        config = json.loads((ROOT / "configs/retrain/mamma_regular.json").read_text())
        loss = active_task_loss(prediction, batch, config)
        self.assertAlmostEqual(float(loss["parts"]["coordinate"]), 21.25, places=5)
        self.assertAlmostEqual(float(loss["parts"]["visibility"]), 0.514216125, places=6)
        self.assertAlmostEqual(float(loss["loss"]), 21.764215469, places=5)
        self.assertFalse(loss["skip_optimizer"])

        batch["coordinate_loss_valid"][0, 2] = True
        prediction["visibility"][0, 2, 0] = -200
        with_unknown = active_task_loss(prediction, batch, config)
        self.assertEqual(with_unknown["counts"]["coordinate"]["points"], 3)
        self.assertEqual(with_unknown["counts"]["visibility"]["points"], 2)
        prediction["visibility"][0, 2, 0] = 200
        self.assertEqual(
            float(active_task_loss(prediction, batch, config)["loss"]), float(with_unknown["loss"])
        )
        batch["coordinate_loss_valid"][:] = False
        batch["visibility_loss_valid"][:] = False
        self.assertTrue(active_task_loss(prediction, batch, config)["skip_optimizer"])

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_embed = torch.nn.Embedding(2, 2)
                self.decoder_detr = torch.nn.Linear(2, 2)
                self.landmarks = torch.nn.Linear(2, 2)
                self.vis_prob = torch.nn.Linear(2, 1)
                self.contact_prob = torch.nn.Linear(2, 1)
                self.floor_contact_prob = torch.nn.Linear(2, 1)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.mask_downscaling = torch.nn.Linear(2, 2)
                self.decoder = Decoder()

        model = FakeModel()
        optimizer, groups = configure_active_optimizer(model, config)
        self.assertEqual(
            [group["name"] for group in groups],
            ["queries", "shared_decoder", "landmarks", "visibility"],
        )
        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertFalse(model.decoder.contact_prob.weight.requires_grad)
        self.assertFalse(model.decoder.floor_contact_prob.weight.requires_grad)
        config["loss_enabled"]["visibility"] = False
        _, groups = configure_active_optimizer(model, config)
        self.assertEqual(
            [group["name"] for group in groups], ["queries", "shared_decoder", "landmarks"]
        )
        self.assertFalse(model.decoder.vis_prob.weight.requires_grad)

    def test_current_resume_rng_sampler_and_contract(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for parameter in model.parameters():
            optimizer.state[parameter] = dict(
                step=torch.tensor(1.0),
                exp_avg=torch.full_like(parameter, 0.2),
                exp_avg_sq=torch.full_like(parameter, 0.04),
            )
        refs = dict.fromkeys(
            (
                "config",
                "train_manifest",
                "validation_manifest",
                "model_config",
                "tensor_state",
                "crop_policy_version",
                "gt_policy",
                "augmentation",
                "training_schema",
            ),
            "fixture",
        )
        refs.update(
            model_config=str(Path("fixture").resolve()),
            tensor_state=str(Path("fixture").resolve()),
            crop_policy_version="mask_bbox_inclusive_v1",
            gt_policy="saved_mamma_512_npz_v1",
            augmentation="off",
            training_schema="mamma_training_v2",
        )
        refs["loss_enabled"] = {"coordinate": True, "visibility": True}
        random.seed(8)
        np.random.seed(8)
        torch.manual_seed(8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            save_regular_boundary(
                path, model, optimizer, 5, capture_rng(), {"epoch": 2, "next_position": 3}, refs
            )
            expected = (random.random(), float(np.random.random()), float(torch.rand(())))
            original = {key: value.clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            other = torch.optim.AdamW(model.parameters(), lr=2e-5)
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(path, model, other, dict(refs, training_schema="old"))
            result = load_regular_boundary(path, model, other, refs)
            self.assertEqual(
                (result["global_update"], result["sampler"]), (5, {"epoch": 2, "next_position": 3})
            )
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original[name]))
            self.assertEqual(len(other.state), len(optimizer.state))
            self.assertEqual(
                (random.random(), float(np.random.random()), float(torch.rand(()))), expected
            )
            evaluation_config = dict(
                model_config="fixture",
                tensor_state="fixture",
                crop_policy_version="mask_bbox_inclusive_v1",
                augmentation="off",
                schema="mamma_training_v2",
                loss_enabled=refs["loss_enabled"],
            )
            load_evaluation_boundary(path, model, evaluation_config)
            with self.assertRaisesRegex(ValueError, "evaluation_checkpoint_contract_mismatch"):
                load_evaluation_boundary(
                    path,
                    model,
                    dict(evaluation_config, loss_enabled={"coordinate": True, "visibility": False}),
                )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["schema"] = "mamma_regular_resume_v2"
            old = Path(temp) / "old.pt"
            torch.save(payload, old)
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(old, model, other, refs)

    def test_execution_still_requires_authorization(self):
        args = type(
            "Args",
            (),
            dict(
                execute=False, output="/tmp/no-run", device="cpu", max_updates=1, checkpoint_every=1
            ),
        )()
        config = json.loads((ROOT / "configs/retrain/mamma_regular.json").read_text())
        with self.assertRaisesRegex(ValueError, "explicit_execute_required"):
            _execution_preflight(args, config, "train")
        args.execute = True
        with self.assertRaisesRegex(ValueError, "configuration_execution_not_authorized"):
            _execution_preflight(args, config, "train")

    def test_train_validation_grouping_is_still_disjoint(self):
        def dataset(episode, asset, manifest):
            row = dict(episode=episode, source_asset_id=asset,
                       source_manifest_id=manifest, status="observed")
            return SimpleNamespace(slots=[row], ready=[0])

        train = dataset("episode_a", "asset_a", "manifest_a")
        validate_disjoint(train, dataset("episode_b", "asset_b", "manifest_b"))
        for other, reason in (
            (dataset("episode_a", "asset_b", "manifest_b"), "episodes"),
            (dataset("episode_b", "asset_a", "manifest_b"), "source_assets"),
            (dataset("episode_b", "asset_b", "manifest_a"), "source_manifests"),
        ):
            with self.assertRaisesRegex(ValueError, "train_validation_group_overlap_" + reason):
                validate_disjoint(train, other)

    def test_evaluation_restores_mask_derived_box_without_model(self):
        expand, affine_for_crop = crop_math()
        box = np.array([100, 100, 140, 180], np.int64)
        center = (box[:2] + box[2:]) / 2
        size = expand((box[2:] - box[:2]).astype(np.float32) * 1.2, (384, 512))
        forward = affine_for_crop(center[0], center[1], size[0], size[1], 384, 512, 1.0, 0)
        import cv2

        item = dict(parent_box_xyxy=box, affine_model_to_erp=cv2.invertAffineTransform(forward))
        model_xy, deployed_xy, inverted_xy = restore_xy(np.array([[0.0, 0.0]]), item, True)
        np.testing.assert_allclose(model_xy, [[192.0, 256.0]], rtol=0, atol=0)
        np.testing.assert_allclose(deployed_xy, [center], rtol=0, atol=1e-5)
        np.testing.assert_allclose(inverted_xy, deployed_xy, rtol=0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
