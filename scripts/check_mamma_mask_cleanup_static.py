"""Static compatibility check; never opens model or optimizer checkpoint."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/acceptance/mamma_mask_derived_crop_cleanup_20261022/static_checks.json'


def main():
    if OUT.exists(): raise RuntimeError('existing_static_checks_refuses_overwrite')
    files=['landmarks/lib/datasets/mamma_roi_pilot.py',
           'landmarks/lib/training/box_probe.py',
           'landmarks/lib/training/policy.py',
           'landmarks/lib/training/recovery.py',
           'landmarks/lib/training/tasks.py',
           'scripts/run_mamma_box_probe.py',
           'scripts/run_mamma_training_smoke.py',
           'scripts/plan_mamma_training_active.py',
           'scripts/check_mamma_mask_crop_cleanup.py']
    for path in files: ast.parse((ROOT/path).read_text(),filename=path)
    old=json.loads((ROOT/'configs/retrain/mamma_training_smoke_20261015.json').read_text())
    current=json.loads((ROOT/'configs/retrain/mamma_training_active_20261022.json').read_text())
    recovery=(ROOT/'landmarks/lib/training/recovery.py').read_text()
    assert old['loss_enabled']['visibility'] is True
    assert current['loss_enabled']['visibility'] is False and current['supervised_visibility'] is False
    assert "'box_stream':box_stream" in recovery and "legacy_box_stream_requires_augmentation_off" in recovery
    assert "mamma_infrastructure_resume_v1" in recovery
    assert (ROOT/'outputs/experiments/mamma_training_readiness_box_20261015/infrastructure_smoke_update1_resume.pt').exists()
    report={'syntax_checked_files':files,'historical_probe_retained':True,
            'legacy_checkpoint_schema':'mamma_infrastructure_resume_v1',
            'legacy_box_stream_retained':'augmentation_off_only',
            'legacy_smoke_visibility_loss_enabled':True,
            'active_visibility_loss_enabled':False,
            'checkpoint_bytes_read':0,'model_forwards':0,'optimizer_steps':0,'gpu_calls':0}
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
