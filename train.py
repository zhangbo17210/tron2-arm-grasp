"""
Tron2 Arm Grasping Training Entry Point
Usage:
    python train.py --task=Tron2ArmGrasp --resume
"""

import argparse
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
from datetime import datetime
from configs.tron2_arm_grasp_config import Tron2ArmGraspCfg, Tron2ArmGraspCfgPPO
from task import Tron2ArmGraspTask
from rsl_rl.runners import OnPolicyRunner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tron2 Arm Grasping Training")
    parser.add_argument("--task", type=str, default="Tron2ArmGrasp",
                        help="Task name (default: Tron2ArmGrasp)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="Checkpoint to resume from (-1 = latest)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use (cuda:0 / cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Override max iterations")
    return parser.parse_args()


def main():
    """Main training loop."""
    args = parse_arguments()

    # Override config from args
    if args.max_iterations:
        Tron2ArmGraspCfgPPO.runner.max_iterations = args.max_iterations

    Tron2ArmGraspCfgPPO.runner.resume = args.resume
    Tron2ArmGraspCfgPPO.runner.checkpoint = args.checkpoint

    # Parse device
    device = args.device
    device_type = device.split(":")[0]
    device_id = int(device.split(":")[1]) if ":" in device else 0

    # Create task
    from isaacgym import gymapi

    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UpAxis(gymapi.UpAxis.Z)
    sim_params.dt = Tron2ArmGraspCfg.sim.dt
    sim_params.substeps = Tron2ArmGraspCfg.sim.substeps
    sim_params.physx.use_gpu = device_type == "cuda"
    sim_params.physx.num_threads = Tron2ArmGraspCfg.sim.physx.num_threads
    sim_params.physx.solver_type = Tron2ArmGraspCfg.sim.physx.solver_type

    task = Tron2ArmGraspTask(
        cfg=Tron2ArmGraspCfg(),
        sim_params=sim_params,
        physics_engine=gymapi.PhysXEngine,
        device_type=device_type,
        device_id=device_id,
        headless=args.headless,
    )

    env = task.create_sim()

    # Create PPO runner
    runner = OnPolicyRunner(
        env,
        Tron2ArmGraspCfgPPO(),
        device=device,
    )

    # Resume if requested
    if args.resume:
        runner.load(Tron2ArmGraspCfgPPO.runner.load_run, Tron2ArmGraspCfgPPO.runner.checkpoint)

    # Training loop
    total_iterations = Tron2ArmGraspCfgPPO.runner.max_iterations
    start_time = time.time()

    for it in range(total_iterations):
        # Collect rollouts
        runner.collect_rollouts()

        # Update policy
        runner.learn()

        # Logging
        if it % Tron2ArmGraspCfgPPO.runner.save_interval == 0:
            elapsed = time.time() - start_time
            fps = runner.env.num_envs * runner.env.episode_length_buf.sum().item() / elapsed
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Iter {it}/{total_iterations} | "
                  f"FPS: {fps:.0f} | "
                  f"Episode reward mean: {runner.episode_rewards.mean():.2f}")

            # Save checkpoint
            runner.save(it)

    print("Training complete!")
    runner.save(total_iterations)


if __name__ == "__main__":
    main()
