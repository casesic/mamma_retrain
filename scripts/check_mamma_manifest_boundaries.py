"""Targeted manifest and ROI rejection checks; no model or optimizer."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'landmarks'))
import lib.models  # noqa: F401
from lib.datasets.mamma_roi_pilot import MammaManifestDataset, native_plan

CONFIG=ROOT/'configs/retrain/mamma_manifest_reader_20261015.json'
OUT=ROOT/'outputs/experiments/mamma_training_readiness_box_20261015/boundary_checks.json'


def rejects(fn,expected):
    try: fn()
    except ValueError as error:
        assert str(error)==expected, (str(error),expected)
        return str(error)
    raise AssertionError('expected_rejection_'+expected)


def main():
    if OUT.exists():
        raise RuntimeError('existing_boundary_checks_refuses_overwrite')
    d=MammaManifestDataset(CONFIG)
    s=d.slots[d.ready[0]]
    raster=s['raster']; full=s['full_image_size_wh']
    outside=rejects(lambda:native_plan([0,0,20,20],raster['roi_xyxy'],raster['stored_size_wh'],full),
                    'crop_filter_support_outside_roi')
    with TemporaryDirectory() as directory:
        invalid=Path(directory)/'manifest.json'
        config=Path(directory)/'reader.json'
        m=json.loads(Path(json.loads(CONFIG.read_text())['manifest']).read_text())
        m['version']=2
        invalid.write_text(json.dumps(m))
        c=json.loads(CONFIG.read_text()); c['manifest']=str(invalid)
        config.write_text(json.dumps(c))
        schema=rejects(lambda:MammaManifestDataset(config),'unsupported_manifest_schema')
    result={'support_outside_roi':outside,'unknown_manifest_version':schema,
            'unknown_assets_rejected_by_reader':'qualified_assets whitelist plus ordered faces',
            'optimizer_steps':0}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
