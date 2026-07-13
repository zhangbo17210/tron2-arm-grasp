"""
Legged Gym Utilities
Common utility functions for training and logging.
"""

import os
import torch
import numpy as np
from datetime import datetime


def get_legs_gym_root_dir():
    """Return the root directory of legged_gym."""
    return os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )


def class_to_dict(obj):
    """Convert a class with nested attributes to a dictionary."""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        if isinstance(val, type):
            result[key] = class_to_dict(val)
        elif isinstance(val, list):
            result[key] = [class_to_dict(v) if isinstance(v, type) else v for v in val]
        else:
            result[key] = val
    return result


def update_cfg_from_args(cfg, args):
    """Update configuration from command line arguments."""
    if hasattr(args, "max_iterations") and args.max_iterations is not None:
        cfg.runner.max_iterations = args.max_iterations
    if hasattr(args, "resume") and args.resume:
        cfg.runner.resume = True
    if hasattr(args, "checkpoint") and args.checkpoint != -1:
        cfg.runner.checkpoint = args.checkpoint
    return cfg
