"""
Isaac Gym Training Logger
TensorBoard-compatible logging for training metrics.
"""

import os
import torch
import numpy as np
from datetime import datetime


class Logger:
    """Simple logger for training metrics."""

    def __init__(self, log_dir, experiment_name):
        self.log_dir = os.path.join(log_dir, experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = None
        self._init_tensorboard()

    def _init_tensorboard(self):
        """Initialize TensorBoard writer if available."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available, logging to CSV only")

    def log(self, key, value, step):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(key, value, step)

    def log_scalar(self, key, value, step):
        """Log scalar (alias for log)."""
        self.log(key, value, step)

    def close(self):
        """Close the logger."""
        if self.writer:
            self.writer.close()
