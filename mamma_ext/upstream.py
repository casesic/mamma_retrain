"""Load the two upstream crop math functions without importing upstream package initializers.

The project-owned adapter remains downstream of the cloned MAMMA source. This
loader executes only the fixed, local utils_eval.py file from this checkout.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np


DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])


@lru_cache(maxsize=1)
def crop_math():
    source = Path(__file__).resolve().parents[1] / 'landmarks/lib/datasets/utils_eval.py'
    spec = importlib.util.spec_from_file_location('_mamma_ext_upstream_utils_eval', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('upstream_crop_math_unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expand_to_aspect_ratio, module.gen_trans_from_patch_cv
