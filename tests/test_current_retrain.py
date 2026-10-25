"""CPU checks for the one current MAMMA data, task and resume contract."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from mamma_ext.data import MammaSavedGTDataset
from mamma_ext.policy import (
    active_task_loss,
    configure_active_optimizer,
    load_active_policy,
)
from mamma_ext.model import restore_xy, summarize_errors
from mamma_ext.data import periodic_mask_box, native_plan, _crop_raster, real_erp_input
from mamma_ext.recovery import (
    capture_rng,
    load_evaluation_boundary,
    load_frozen_20261024_point_mean_state,
    load_regular_boundary,
    save_regular_boundary,
)
from mamma_ext.runner import _execution_preflight, _tensor_batch, mixed_order
from mamma_ext.sampler import (make_occlusion_plan, load_occlusion_plan,
                               make_one_epoch_plan, load_one_epoch_plan)
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
        observation_id="fixture_target",
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
    return bundle, label, annotation


class CurrentRetrainTest(unittest.TestCase):
    def test_one_epoch_plan_real_batch_and_resume_contract(self):
        slots = [{"observation_id": f"obs_{i}", "episode": f"episode_{i % 3}",
                  "person_id": i % 2 + 1, "frame_index": i} for i in range(9)]
        class Fixture:
            def __len__(self):
                return len(self.ready)
        dataset = Fixture()
        dataset.slots, dataset.ready = slots, list(range(9))
        dataset.cfg = {"manifests": ["train_fixture"]}
        with patch("mamma_ext.sampler.label_counts", side_effect=lambda _, i: {
            "coordinates": 0 if i == 8 else 3,
            "positive": 0 if i in (7, 8) else 2,
            "negative": 1 if i == 0 else 0,
        }):
            plan = make_one_epoch_plan(dataset, 20261024)
            self.assertEqual(len(plan["entries"]), 8)
            self.assertEqual([r["observation_id"] for r in plan["excluded_no_supervision"]], ["obs_8"])
            self.assertEqual(len(set(row["observation_id"] for row in plan["entries"])), 8)
            self.assertEqual(plan, make_one_epoch_plan(dataset, 20261024))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "plan.json"
                write_json(path, plan)
                self.assertEqual(load_one_epoch_plan(path, dataset, 20261024), plan)
                rows = plan["entries"]
                batches = [rows[i:i + 3] for i in range(0, len(rows), 3)]
                self.assertEqual([len(batch) for batch in batches], [3, 3, 2])
                self.assertEqual([r["observation_id"] for b in batches for r in b],
                                 [r["observation_id"] for r in rows])

                refs = {key: "fixture" for key in (
                    "config", "model_config", "tensor_state", "crop_policy_version",
                    "gt_policy", "augmentation", "training_schema", "visibility_reduction",
                    "sampling_plan_id")}
                refs.update(train_manifests=["train_fixture"], validation_manifests=["val_fixture"],
                            model_config=str(Path("fixture").resolve()),
                            tensor_state=str(Path("fixture").resolve()),
                            gt_policy="saved_mamma_512_npz_v1",
                            sampling_plan_id=plan["plan_id"],
                            loss_enabled={"coordinate": True, "visibility": True},
                            optimizer={"type": "AdamW"}, sampler_seed=20261024,
                            sampling={"mode": "one_epoch_v1", "plan": str(path)},
                            training_schema="mamma_training_v7", batch_size=3)
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.AdamW(model.parameters())
                checkpoint = Path(temp) / "checkpoint.pt"
                save_regular_boundary(checkpoint, model, optimizer, 2, capture_rng(),
                                      {"seed": 20261024, "epoch": 0, "next_position": 6}, refs)
                restored = load_regular_boundary(checkpoint, model, optimizer, refs)
                self.assertEqual((restored["global_update"], restored["sampler"]["next_position"]), (2, 6))
                with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                    load_regular_boundary(checkpoint, model, optimizer, dict(refs, batch_size=4))
                evaluation_config = {
                    "model_config": refs["model_config"], "tensor_state": refs["tensor_state"],
                    "crop_policy_version": refs["crop_policy_version"],
                    "augmentation": refs["augmentation"], "schema": refs["training_schema"],
                    "loss_enabled": refs["loss_enabled"],
                    "visibility_reduction": refs["visibility_reduction"],
                    "optimizer": refs["optimizer"], "sampler_seed": refs["sampler_seed"],
                    "sampling": refs["sampling"], "batch_size": 3,
                }
                load_evaluation_boundary(checkpoint, model, evaluation_config, ["val_fixture"])
                with self.assertRaisesRegex(ValueError, "evaluation_checkpoint_contract_mismatch"):
                    load_evaluation_boundary(checkpoint, model,
                                             dict(evaluation_config, batch_size=4), ["val_fixture"])

        item = {"img": np.zeros((3, 2, 2), np.float32), "mask": np.zeros((1, 2, 2), np.float32),
                "xy_target": np.zeros((2, 2), np.float32),
                "coordinate_loss_valid": np.array([True, False]),
                "visibility": np.array([1, -1]),
                "visibility_loss_valid": np.array([True, False])}
        batch = _tensor_batch([item, item, item], torch.device("cpu"))
        self.assertEqual(tuple(batch["img"].shape), (3, 3, 2, 2))
        self.assertEqual(tuple(batch["visibility"].shape), (3, 2))

    def test_batch_loss_uses_task_specific_sample_means(self):
        from mamma_ext.tasks import masked_task_loss
        prediction = {"joints2d": torch.tensor([[[.1, 0., 0.], [.2, 0., 0.]],
                                                   [[.3, 0., 0.], [.4, 0., 0.]]]),
                      "visibility": torch.tensor([[[.5], [-.5]], [[.8], [.9]]])}
        batch = {"xy_target": torch.zeros((2, 2, 2)),
                 "coordinate_loss_valid": torch.tensor([[True, True], [True, False]]),
                 "visibility": torch.tensor([[1, 0], [-1, -1]]),
                 "visibility_loss_valid": torch.tensor([[True, True], [False, False]])}
        config = json.loads((ROOT / "configs/retrain/mamma_city_merged_one_epoch_20261025.json").read_text())
        whole = active_task_loss(prediction, batch, config)
        singles = [{key: value[i:i + 1] for key, value in batch.items()} for i in range(2)]
        pred_singles = [{key: value[i:i + 1] for key, value in prediction.items()} for i in range(2)]
        losses = [active_task_loss(pred_singles[i], singles[i], config) for i in range(2)]
        self.assertTrue(torch.allclose(whole["parts"]["coordinate"],
                                       (losses[0]["parts"]["coordinate"] + losses[1]["parts"]["coordinate"]) / 2))
        self.assertTrue(torch.allclose(whole["parts"]["visibility"], losses[0]["parts"]["visibility"]))

    def test_one_epoch_policy_rejects_missing_or_invalid_batch(self):
        source = ROOT / "configs/retrain/mamma_city_merged_one_epoch_20261025.json"
        config, dataset = load_active_policy(source, "validation")
        self.assertEqual((config["batch_size"], len(dataset)), (4, 435))
        with tempfile.TemporaryDirectory(dir=source.parent) as temp:
            path = Path(temp) / "invalid.json"
            for bad in (0, -1, True, "4"):
                altered = json.loads(source.read_text())
                altered["batch_size"] = bad
                write_json(path, altered)
                with self.assertRaisesRegex(ValueError, "invalid_training_batch_size"):
                    load_active_policy(path)
            altered = json.loads(source.read_text())
            del altered["batch_size"]
            write_json(path, altered)
            with self.assertRaisesRegex(ValueError, "unsupported_active_training_policy"):
                load_active_policy(path)

    def test_fixed_occlusion_quota_plan_and_contract(self):
        slots = [{"observation_id": f"sample_{i}", "episode": f"episode_{i % 4}",
                  "person_id": i % 3 + 1, "frame_index": i // 12}
                 for i in range(526)]
        class FixtureDataset:
            def __len__(self):
                return len(self.ready)
        dataset = FixtureDataset()
        dataset.slots = slots
        dataset.ready = list(range(526))
        dataset.cfg = {"manifests": ["train_fixture"]}
        def counts(_dataset, index):
            return {"positive": 4, "negative": 2 if index < 126 else 0,
                    "coordinates": 8}
        with patch("mamma_ext.sampler.label_counts", side_effect=counts):
            before = np.random.get_state()
            plan = make_occlusion_plan(dataset, 20261024)
            after = np.random.get_state()
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(plan, make_occlusion_plan(dataset, 20261024))
            entries = plan["entries"]
            self.assertEqual(len(entries), 500)
            self.assertTrue(all(sum(row["pool"] == "occlusion" for row in entries[i:i+10]) == 3
                                for i in range(0, 500, 10)))
            occluded = [row["observation_id"] for row in entries
                        if row["pool"] == "occlusion"]
            self.assertEqual(len(set(occluded[:126])), 126)
            self.assertEqual(len(set(row["observation_id"] for row in entries
                                     if row["pool"] == "other")), 350)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "plan.json"
                write_json(path, plan)
                self.assertEqual(load_occlusion_plan(path, dataset, 20261024), plan)
                changed = dict(plan, plan_id="wrong")
                write_json(path, changed)
                with self.assertRaisesRegex(ValueError, "sampling_plan_identity_or_data_mismatch"):
                    load_occlusion_plan(path, dataset, 20261024)
    def test_bundle_identity_time_template_and_files_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle, label, annotation = tiny_bundle(root)
            reader = [root / "observations.json"]
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
        current, dataset = load_active_policy(config, split="validation")
        self.assertEqual(current["schema"], "mamma_training_v6")
        self.assertEqual(current["input_config"], str(config.parent / "city_full_20261024_split.json"))
        self.assertEqual(len(dataset.cfg["manifests"]), 4)
        self.assertFalse(current["execution_authorized"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "old.json"
            old = json.loads(config.read_text())
            old["schema"] = "mamma_training_v1"
            old["supervised_visibility"] = True
            write_json(path, old)
            with self.assertRaisesRegex(ValueError, "unsupported_active_training_policy"):
                load_active_policy(path)

            for invalid in (None, "weighted_bce", 1):
                malformed = json.loads(config.read_text())
                malformed["visibility_reduction"] = invalid
                write_json(path, malformed)
                with self.assertRaisesRegex(ValueError, "unsupported_active_training_policy"):
                    load_active_policy(path)
            malformed = json.loads(config.read_text())
            del malformed["visibility_reduction"]
            write_json(path, malformed)
            with self.assertRaisesRegex(ValueError, "unsupported_active_training_policy"):
                load_active_policy(path)
            malformed = json.loads(config.read_text())
            malformed["sampling"] = {"mode": "occlusion_quota_v1", "total_updates": 500,
                                     "occlusion_updates": 151, "plan": "schedule.json"}
            write_json(path, malformed)
            with self.assertRaisesRegex(ValueError, "unsupported_sampling_policy"):
                load_active_policy(path)

            for task in ("coordinate", "visibility"):
                for invalid in (0, 1, "true", None):
                    malformed = json.loads(config.read_text())
                    malformed["loss_enabled"][task] = invalid
                    write_json(path, malformed)
                    with self.assertRaisesRegex(ValueError, "task_loss_optimizer_mismatch"):
                        load_active_policy(path)

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

    def test_visibility_class_mean_formula_gradient_and_single_class(self):
        from mamma_ext.tasks import masked_task_loss

        logits = torch.tensor([[[0.5], [-0.7], [1.2], [9.0]]], requires_grad=True)
        prediction = {"joints2d": torch.zeros((1, 4, 3)), "visibility": logits}
        batch = {
            "xy_target": torch.zeros((1, 4, 2)),
            "coordinate_loss_valid": torch.zeros((1, 4), dtype=torch.bool),
            "visibility": torch.tensor([[1, 0, 1, -1]]),
            "visibility_loss_valid": torch.tensor([[True, True, True, False]]),
        }
        enabled = {"coordinate": False, "visibility": True}
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[0, :3, 0], torch.tensor([1., 0., 1.]), reduction="none")
        ordinary = masked_task_loss(prediction, batch, enabled, visibility_reduction="point_mean")
        balanced = masked_task_loss(prediction, batch, enabled, visibility_reduction="class_mean")
        self.assertTrue(torch.allclose(ordinary["loss"], bce.mean()))
        self.assertTrue(torch.allclose(balanced["loss"],
                                       .5 * bce[[0, 2]].mean() + .5 * bce[1]))
        same_positive_logits = logits.detach().clone()
        same_positive_logits[0, 2] = same_positive_logits[0, 0]
        same_positive = {**prediction, "visibility": same_positive_logits}
        duplicate = {**same_positive, "visibility": torch.cat((same_positive_logits,
                                                                  same_positive_logits[:, :1]), dim=1)}
        duplicate_batch = {**batch,
                           "visibility": torch.tensor([[1, 0, 1, -1, 1]]),
                           "visibility_loss_valid": torch.tensor([[True, True, True, False, True]]),
                           "coordinate_loss_valid": torch.zeros((1, 5), dtype=torch.bool)}
        duplicate["joints2d"] = torch.zeros((1, 5, 3))
        duplicate_batch["xy_target"] = torch.zeros((1, 5, 2))
        self.assertTrue(torch.allclose(
            masked_task_loss(duplicate, duplicate_batch, enabled,
                             visibility_reduction="class_mean")["loss"],
            masked_task_loss(same_positive, batch, enabled,
                             visibility_reduction="class_mean")["loss"]))
        balanced["loss"].backward()
        self.assertAlmostEqual(float(logits.grad[0, 0, 0] + logits.grad[0, 2, 0]),
                               float(torch.sigmoid(logits[0, 0, 0]) / 4 +
                                     torch.sigmoid(logits[0, 2, 0]) / 4 - .5), places=6)
        self.assertEqual(float(logits.grad[0, 3, 0]), 0.)
        batch["visibility_loss_valid"][0, 1] = False
        single = masked_task_loss(prediction, batch, enabled, visibility_reduction="class_mean")
        self.assertTrue(torch.allclose(single["loss"], bce[[0, 2]].mean()))
        batch["visibility_loss_valid"][:] = False
        self.assertTrue(masked_task_loss(prediction, batch, enabled,
                                         visibility_reduction="class_mean")["skip_optimizer"])

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
                "train_manifests",
                "validation_manifests",
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
            crop_policy_version="mask_bbox_periodic_v2",
            gt_policy="saved_mamma_512_npz_v1",
            augmentation="off",
            training_schema="mamma_training_v6",
        )
        refs["loss_enabled"] = {"coordinate": True, "visibility": True}
        refs["visibility_reduction"] = "class_mean"
        refs["optimizer"] = {"type": "AdamW"}
        refs["sampler_seed"] = 8
        refs["sampling"] = {"mode": "episode_round_robin"}
        refs["sampling_plan_id"] = "episode_round_robin"
        refs["train_manifests"] = ["fixture_train"]
        refs["validation_manifests"] = ["fixture_validation"]
        random.seed(8)
        np.random.seed(8)
        torch.manual_seed(8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.pt"
            save_regular_boundary(
                path, model, optimizer, 5, capture_rng(), {"seed": 8, "epoch": 2, "next_position": 3}, refs
            )
            expected = (random.random(), float(np.random.random()), float(torch.rand(())))
            original = {key: value.clone() for key, value in model.state_dict().items()}
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            other = torch.optim.AdamW(model.parameters(), lr=2e-5)
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(path, model, other, dict(refs, training_schema="old"))
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(path, model, other, dict(refs, visibility_reduction="point_mean"))
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(path, model, other,
                                      dict(refs, sampling_plan_id="different_plan"))
            result = load_regular_boundary(path, model, other, refs)
            self.assertEqual(
                (result["global_update"], result["sampler"]), (5, {"seed": 8, "epoch": 2, "next_position": 3})
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
                crop_policy_version="mask_bbox_periodic_v2",
                augmentation="off",
                schema="mamma_training_v6",
                loss_enabled=refs["loss_enabled"],
                visibility_reduction="class_mean",
                optimizer=refs["optimizer"],
                sampler_seed=8,
                sampling={"mode": "episode_round_robin"},
            )
            load_evaluation_boundary(path, model, evaluation_config, ["fixture_validation"])
            with self.assertRaisesRegex(ValueError, "evaluation_checkpoint_contract_mismatch"):
                load_evaluation_boundary(
                    path,
                    model,
                    dict(evaluation_config, loss_enabled={"coordinate": True, "visibility": False}),
                    ["fixture_validation"],
                )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["schema"] = "mamma_regular_resume_v2"
            old = Path(temp) / "old.pt"
            torch.save(payload, old)
            with self.assertRaisesRegex(ValueError, "regular_resume_schema_or_refs_mismatch"):
                load_regular_boundary(old, model, other, refs)
            payload["schema"] = "mamma_regular_resume_v5"
            payload["global_update"] = 500
            payload["refs"]["training_schema"] = "mamma_training_v4"
            del payload["refs"]["visibility_reduction"]
            torch.save(payload, old)
            load_frozen_20261024_point_mean_state(
                old, model, dict(evaluation_config, visibility_reduction="point_mean"),
                ["fixture_validation"])
            with self.assertRaisesRegex(ValueError, "frozen_point_mean_model_contract_mismatch"):
                load_frozen_20261024_point_mean_state(
                    old, model, evaluation_config, ["fixture_validation"])

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
        validate_disjoint(train, dataset("episode_b", "asset_a", "manifest_a"))
        with self.assertRaisesRegex(ValueError, "train_validation_group_overlap_episodes"):
            validate_disjoint(train, dataset("episode_a", "asset_b", "manifest_b"))
        validation = dataset("episode_b", "asset_b", "manifest_b")
        validation.slots.append(dict(episode="episode_a", status="absent", absent_reason="empty_target_mask"))
        with self.assertRaisesRegex(ValueError, "train_validation_group_overlap_episodes"):
            validate_disjoint(train, validation)

    def test_multibundle_and_mixed_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = root / "a"
            b = root / "b"
            a.mkdir(); b.mkdir()
            first, _, _ = tiny_bundle(a)
            second, _, _ = tiny_bundle(b)
            second["episode"] = "fixture_b"
            second["observations"][0]["episode"] = "fixture_b"
            second["observations"][0]["observation_id"] = "fixture_b_target"
            with np.load(b / "labels.npz") as labels:
                payload = {key: labels[key] for key in labels.files}
            payload["scene_episode_id"] = np.array("fixture_b")
            np.savez(b / "labels.npz", **payload)
            with np.load(b / "annotations.npz") as annotations:
                payload = {key: annotations[key] for key in annotations.files}
            payload["scene_episode_id"] = np.array("fixture_b")
            np.savez(b / "annotations.npz", **payload)
            write_json(b / "observations.json", second)
            dataset = MammaSavedGTDataset([a / "observations.json", b / "observations.json"])
            self.assertEqual(len(dataset), 2)
            self.assertEqual(set(mixed_order(dataset, 9, 0)), {0, 1})
            self.assertEqual(mixed_order(dataset, 9, 0), mixed_order(dataset, 9, 0))
            class Fake:
                slots = [{"episode": f"scene_{i}"} for i in range(16) for _ in range(3)]
                ready = list(range(48))

                def __len__(self):
                    return len(self.ready)

            fake = Fake()
            order = mixed_order(fake, 20261024, 0)
            self.assertEqual(len(set(fake.slots[i]["episode"] for i in order[:16])), 16)
            self.assertEqual(len(set(order)), 48)
            self.assertEqual(order[25], mixed_order(fake, 20261024, 0)[25])
            with self.assertRaisesRegex(ValueError, "duplicate_data_package"):
                MammaSavedGTDataset([a / "observations.json"] * 2)
            write_json(b / "observations.json", first)
            with self.assertRaisesRegex(ValueError, "duplicate_episode_package"):
                MammaSavedGTDataset([a / "observations.json", b / "observations.json"])

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

    def test_periodic_mask_crop_and_erp_restoration(self):
        import cv2

        width = 64
        mask = np.zeros((48, width), np.uint8)
        mask[12:34, :7] = 255
        mask[12:34, 58:] = 255
        box = periodic_mask_box(mask, np.array([0, 8]), width)
        np.testing.assert_array_equal(box, [58, 20, 70, 41])
        roi = np.array([0, 8, width, 56])
        center, size, _, rect = native_plan(box, roi, [64, 48], [64, 64], periodic_x=True)
        self.assertLess(rect[2] - rect[0], width)
        _, affine_for_crop = crop_math()
        forward = affine_for_crop(center[0], center[1], size[0], size[1], 384, 512, 1.0, 0)
        item = dict(
            parent_box_xyxy=box,
            affine_model_to_erp=cv2.invertAffineTransform(forward),
            periodic_x=True,
            full_size_wh=np.array([width, 64]),
        )
        point = np.array([[2.0, 30.0]])
        unfolded = point.copy()
        unfolded[:, 0] += width
        model_xy = unfolded @ forward[:, :2].T + forward[:, 2]
        raw = model_xy / [384, 512] * 2 - 1
        _, deployed, inverted = restore_xy(raw, item, True)
        np.testing.assert_allclose(deployed, point, atol=1e-4)
        np.testing.assert_allclose(inverted, point, atol=1e-4)
        errors, _ = summarize_errors(np.array([[63.5, 30.0]]), np.array([[0.5, 30.0]]), [True], width)
        self.assertAlmostEqual(float(errors[0]), 1.0)

        image = np.zeros((48, width, 3), np.uint8)
        image[:, :, 0] = np.arange(width, dtype=np.uint8)
        image[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
        def support_for(target_mask):
            target_box = periodic_mask_box(target_mask, roi[:2], width)
            _, _, _, target_rect = native_plan(target_box, roi, [64, 48], [64, 64], periodic_x=True)
            return dict(crop_policy_version="mask_bbox_periodic_v2",
                        box_xyxy=target_box.tolist(),
                        sampling_rect_erp=target_rect.tolist(),
                        source_sampling_supported=True)
        seam_rgb, seam_mask, _, _ = _crop_raster(
            image, mask, roi, np.array([64, 64]), support_for(mask)
        )
        centered_image = np.roll(image, 32, axis=1)
        centered_mask = np.roll(mask, 32, axis=1)
        center_rgb, center_mask, _, _ = _crop_raster(
            centered_image, centered_mask, roi, np.array([64, 64]), support_for(centered_mask)
        )
        np.testing.assert_allclose(seam_rgb, center_rgb, atol=1e-5)
        np.testing.assert_array_equal(seam_mask, center_mask)

    def test_native_crop_support_matches_float32_export_at_exact_aspect(self):
        from mamma_ext.data import mask_box_inclusive

        roi = np.array([0, 0, 180, 180])
        mask = np.zeros((180, 180), np.uint8)
        mask[41:132, 40:82] = 255
        box = mask_box_inclusive(mask, roi[:2])
        np.testing.assert_array_equal(box, [40, 41, 81, 131])
        full = np.array([200, 180])
        _, _, _, rect = native_plan(box, roi, [180, 180], full)
        np.testing.assert_array_equal(rect, [16, 28, 104, 143])
        image = np.zeros((180, 180, 3), np.uint8)
        support = dict(
            crop_policy_version="mask_bbox_inclusive_v1",
            box_xyxy=box.tolist(),
            sampling_rect_erp=rect.tolist(),
            source_sampling_supported=True,
        )
        rgb, cropped_mask, _, used_box = _crop_raster(image, mask, roi, full, support)
        self.assertEqual(rgb.shape, (3, 512, 384))
        self.assertEqual(cropped_mask.shape, (512, 384))
        np.testing.assert_array_equal(used_box, box)

    def test_real_erp_vertical_black_pad_and_strict_bundle_unchanged(self):
        import cv2

        height, width = 1024, 2048
        image = np.zeros((height, width, 3), np.uint8)
        top = np.zeros((height, width), np.uint8)
        top[:100, :80] = 255
        top[:100, -70:] = 255
        image[900:, :, 2] = 255  # A wrong negative slice would import this red edge.
        upper = real_erp_input(image, top)
        self.assertGreater(upper["vertical_padding_top_bottom"][0], 0)
        self.assertEqual(upper["vertical_padding_top_bottom"][1], 0)
        self.assertTrue(np.isfinite(upper["img"]).all())
        self.assertGreater(int(upper["mask"].sum()), 0)
        from mamma_ext.upstream import DEFAULT_MEAN, DEFAULT_STD
        self.assertAlmostEqual(float(upper["img"][0].max()),
                               float(-DEFAULT_MEAN[0] / DEFAULT_STD[0]), places=5)
        with self.assertRaisesRegex(ValueError, "crop_filter_support_outside_roi"):
            _crop_raster(image, top, np.array([0, 0, width, height]),
                         np.array([width, height]), periodic_x=True)
        shifted = real_erp_input(np.roll(image, 700, axis=1), np.roll(top, 700, axis=1))
        np.testing.assert_array_equal(upper["img"], shifted["img"])
        np.testing.assert_array_equal(upper["mask"], shifted["mask"])
        raw = np.zeros((512, 2), np.float32)
        _, deployed, inverted = restore_xy(raw, upper)
        np.testing.assert_allclose(deployed, inverted, atol=0.01)
        assert np.isfinite(deployed).all()

        bottom = np.zeros_like(top)
        bottom[-100:, 800:950] = 255
        lower = real_erp_input(image, bottom)
        self.assertGreater(lower["vertical_padding_top_bottom"][1], 0)
        self.assertEqual(lower["vertical_padding_top_bottom"][0], 0)
        boundary_model = np.array([[[192., 511.]]], np.float32)
        erp = cv2.transform(boundary_model, lower["affine_model_to_erp"])[0, 0]
        self.assertGreater(erp[1], height)  # Do not clamp an outside prediction.

        middle = np.zeros_like(top)
        middle[380:650, 800:950] = 255
        ordinary = real_erp_input(image, middle)
        old_rgb, old_mask, old_affine, old_box = _crop_raster(
            image, middle, np.array([0, 0, width, height]),
            np.array([width, height]), periodic_x=True
        )
        np.testing.assert_array_equal(ordinary["img"], old_rgb)
        np.testing.assert_array_equal(ordinary["mask"], old_mask[None].astype(np.float32) / 255)
        np.testing.assert_array_equal(ordinary["affine_erp_to_model"], old_affine)
        np.testing.assert_array_equal(ordinary["parent_box_xyxy"], old_box)


if __name__ == "__main__":
    unittest.main()
