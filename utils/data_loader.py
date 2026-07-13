"""
Data Loading Utilities
Load and preprocess demonstration data for imitation learning.
"""

import os
import numpy as np
import torch
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_demo_episodes(filename=None):
    """
    Load demonstration episodes from data/raw/.
    
    Returns:
        dict with keys: 'obs', 'actions', 'joint_pos', 'joint_vel'
    """
    raw_dir = DATA_DIR / "raw"
    
    if filename is None:
        # Find first available .npy file
        npy_files = list(raw_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {raw_dir}")
        filename = npy_files[0].name
    
    filepath = raw_dir / filename
    print(f"[DATA] Loading demonstrations from {filepath}")
    data = np.load(filepath, allow_pickle=True)
    
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = data.item()  # dict saved as array
    
    return data


def get_data_summary():
    """Print summary of available data files."""
    raw_dir = DATA_DIR / "raw"
    files = list(raw_dir.iterdir()) if raw_dir.exists() else []
    
    print(f"\n[DATA] Available data files in {raw_dir}:")
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.1f} MB)")
    
    if not files:
        print("  (empty)")
    print()


if __name__ == "__main__":
    get_data_summary()
