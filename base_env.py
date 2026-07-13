"""
Base leg gym environment class.
Provides common infrastructure for Isaac Gym RL environments.
"""

import os
import numpy as np
import torch
import isaacgym
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    torch_rand_float,
    get_euler_xyz,
    quat_rotate_inverse,
    to_torch,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
)


class BaseEnv:
    """Base environment providing common setup and utilities."""

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.gym = gymapi.acquire_gym()
        self.device_type = device_type
        self.device_id = device_id
        self.headless = headless
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # Determine device
        if device_type == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.num_envs = cfg.env.num_envs
        self.num_actions = cfg.env.num_actions
        self.num_observations = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs

        # Episode management
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Reward tracking
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.episode_sums = {}

        # Create sim
        self.create_sim()

        # Setup after sim creation
        self._setup_env()

    def create_sim(self):
        """Create the simulation."""
        sim_params = self.sim_params
        sim_params.physx.use_gpu = self.device_type == "cuda"
        sim_params.up_axis = gymapi.UpAxis(self.cfg.sim.up_axis)
        sim_params.dt = self.cfg.sim.dt
        sim_params.substeps = self.cfg.sim.substeps
        sim_params.physx.num_threads = self.cfg.sim.physx.num_threads
        sim_params.physx.solver_type = self.cfg.sim.physx.solver_type
        sim_params.physx.num_position_iterations = self.cfg.sim.physx.num_position_iterations
        sim_params.physx.num_velocity_iterations = self.cfg.sim.physx.num_velocity_iterations
        sim_params.physx.contact_offset = self.cfg.sim.physx.contact_offset
        sim_params.physx.rest_offset = self.cfg.sim.physx.rest_offset

        self.sim = self.gym.create_sim(
            self.device_id, self.device_id, self.physics_engine, sim_params
        )

        # Create ground plane
        self._create_ground_plane()

    def _create_ground_plane(self):
        """Create the ground plane."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _setup_env(self):
        """Setup environments - override in subclass."""
        raise NotImplementedError

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        raise NotImplementedError

    def step(self, actions):
        """Execute one simulation step."""
        raise NotImplementedError

    def compute_observations(self):
        """Compute observations for all environments."""
        raise NotImplementedError

    def compute_reward(self):
        """Compute rewards for all environments."""
        raise NotImplementedError
