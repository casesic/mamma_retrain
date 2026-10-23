"""Historical read-only plan CLI; delegates to the regular extension."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mamma_ext.runner import read_plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/retrain/mamma_training_simple_20261022.json'))
    args=parser.parse_args()
    _,dataset,plan=read_plan(args.config)
    plan.update(ready_count=len(dataset),
                ready_count_meaning='observed manifest slots; no full image or label qualification sweep',
                slots=[dataset.slot_status(i) for i in range(len(dataset.slots))])
    print(json.dumps(plan,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
