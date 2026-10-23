"""Thin CLI for project-owned MAMMA plan, data check, train and evaluation."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mamma_ext.runner import main


if __name__ == '__main__':
    main()
