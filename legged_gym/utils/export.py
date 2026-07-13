"""
Training Utilities
"""

import os
import glob
from datetime import datetime


def get_latest_run(log_dir):
    """Get the most recent training run directory."""
    runs = sorted(glob.glob(os.path.join(log_dir, "*")))
    if not runs:
        return None
    return runs[-1]


def get_latest_checkpoint(log_dir, run_name=None):
    """Get the latest checkpoint from a run."""
    if run_name is None:
        run_name = get_latest_run(log_dir)
    if run_name is None:
        return -1, -1

    checkpoints = sorted(glob.glob(os.path.join(run_name, "model_*.pt")))
    if not checkpoints:
        return -1, -1

    latest = checkpoints[-1]
    iteration = int(os.path.basename(latest).split("_")[1].split(".")[0])
    return latest, iteration
