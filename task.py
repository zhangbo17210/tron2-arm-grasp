"""
Tron2 Arm Grasping - Gym Task Registration
Registers the task with Isaac Gym's task factory.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaacgym import gymapi
from base_env import BaseEnv
from env import Tron2ArmEnv
from configs.tron2_arm_grasp_config import Tron2ArmGraspCfg, Tron2ArmGraspCfgPPO


class Tron2ArmGraspTask:
    """Task wrapper for Tron2 arm grasping."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless

    def create_sim(self):
        """Create the environment."""
        self.env = Tron2ArmEnv(
            cfg=self.cfg.env,
            sim_params=self.sim_params,
            physics_engine=self.physics_engine,
            device_type=self.device_type,
            device_id=self.device_id,
            headless=self.headless,
        )
        return self.env

    def get_cppo_config(self):
        """Return PPO configuration."""
        return Tron2ArmGraspCfgPPO
