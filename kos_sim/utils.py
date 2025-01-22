"""Defines some utility functions."""

import os
from pathlib import Path


def get_sim_artifacts_path() -> Path:
    base_path = os.getenv("KOS_SIM_CACHE_PATH", ".kos-sim")
    return Path(base_path).expanduser().resolve()
