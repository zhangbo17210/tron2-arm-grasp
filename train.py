"""
Tron2 Arm Grasping Training Entry Point
Run with: gm-run train.py --task=Tron2ArmGrasp --headless

IMPORTANT: isaacgym MUST be imported before torch!
Isaac Gym's gymdeps checks sys.modules for torch and raises ImportError
if torch was loaded first.
"""

import argparse
import sys
import os

# Ensure repo root is in path for sub-module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ISAAC GYM FIRST (must be before torch!) ────────────────────────
from isaacgym import gymapi

# ── Standard libs ──────────────────────────────────────────────────
import time
from datetime import datetime

# ── Third-party libs (after isaacgym) ──────────────────────────────
import torch
import numpy as np

# ── Config imports ─────────────────────────────────────────────────
from configs.tron2_arm_grasp_config import Tron2ArmGraspCfg, Tron2ArmGraspCfgPPO
from task import Tron2ArmGraspTask
from rsl_rl.runners import OnPolicyRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Tron2 Arm Grasping Training")
    parser.add_argument("--task", type=str, default="Tron2ArmGrasp")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=int, default=-1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_iterations", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_iterations:
        Tron2ArmGraspCfgPPO.runner.max_iterations = args.max_iterations
    Tron2ArmGraspCfgPPO.runner.resume = args.resume
    Tron2ArmGraspCfgPPO.runner.checkpoint = args.checkpoint

    device = args.device
    device_type = device.split(":")[0]
    device_id = int(device.split(":")[1]) if ":" in device else 0

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis(1)  # 1=Y-up (matches config up_axis=1)
    sim_params.dt = Tron2ArmGraspCfg.sim.dt
    sim_params.substeps = Tron2ArmGraspCfg.sim.substeps
    sim_params.physx.use_gpu = device_type == "cuda"
    sim_params.physx.num_threads = Tron2ArmGraspCfg.sim.physx.num_threads
    sim_params.physx.solver_type = Tron2ArmGraspCfg.sim.physx.solver_type

    print(f"[INFO] Creating Tron2ArmGrasp task on {device}...")
    task = Tron2ArmGraspTask(
        cfg=Tron2ArmGraspCfg(),
        sim_params=sim_params,
        physics_engine=gymapi.PhysXEngine,
        device_type=device_type,
        device_id=device_id,
        headless=args.headless,
    )
    env = task.create_sim()
    print(f"[INFO] Environment created with {env.num_envs} envs")

    runner = OnPolicyRunner(env, Tron2ArmGraspCfgPPO(), device=device)

    if args.resume:
        runner.load(Tron2ArmGraspCfgPPO.runner.load_run,
                    Tron2ArmGraspCfgPPO.runner.checkpoint)

    total = Tron2ArmGraspCfgPPO.runner.max_iterations
    t0 = time.time()

    for it in range(total):
        runner.collect_rollouts()
        runner.learn()

        if it % Tron2ArmGraspCfgPPO.runner.save_interval == 0:
            elapsed = time.time() - t0
            fps = env.num_envs * env.episode_length_buf.sum().item() / elapsed
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iter {it}/{total} | "
                  f"FPS: {fps:.0f} | "
                  f"Reward: {runner.episode_rewards.mean():.2f}")
            runner.save(it)

    print("Training complete!")
    runner.save(total)


if __name__ == "__main__":
    main()
