"""
Tron2 Arm Grasping Environment
Isaac Gym environment for robotic arm grasping manipulation tasks.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# IMPORTANT: isaacgym MUST be imported before torch!
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import (
    torch_rand_float,
    to_torch,
)

import torch
import numpy as np

from base_env import BaseEnv
from configs.tron2_arm_config import Tron2ArmCfg, Tron2ArmCfgPPO

# Setup LEGGED_GYM_ROOT_DIR (repo root)
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Tron2ArmEnv(BaseEnv):
    """Tron2 arm grasping environment."""

    def __init__(self, cfg: Tron2ArmCfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.cfg = cfg

        # Asset handles
        self.actor_handles = []
        self.object_handles = []

        # State buffers (initialized after env setup)
        self.root_state = None
        self.dof_state = None
        self.force_sensor_tensor = None

        # Command targets
        self.commands = torch.zeros(self.num_envs, cfg.commands.num_commands, device=self.device)

    def _setup_env(self):
        """Create environments with robot and objects."""
        cfg = self.cfg
        asset_root = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "robots")
        asset_file = cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        # Load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DofDriveMode(cfg.control.control_type)
        asset_options.collapse_fixed_joints = cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = cfg.asset.fix_base_link
        asset_options.density = cfg.asset.density
        asset_options.angular_damping = cfg.asset.angular_damping
        asset_options.linear_damping = cfg.asset.linear_damping
        asset_options.max_angular_velocity = cfg.asset.max_angular_velocity
        asset_options.armature = cfg.asset.armature

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Load object asset (simple box for grasping)
        obj_asset_options = gymapi.AssetOptions()
        obj_asset_options.density = 200.0
        obj_asset = self.gym.load_asset(self.sim, asset_root, "objects/box.urdf", obj_asset_options)

        # Create environments
        spacing = cfg.env.env_spacing
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.actor_handles = []
        self.object_handles = []

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))

            # Spawn robot arm
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*cfg.init_state.pos)
            pose.r = gymapi.Quat(*cfg.init_state.rot)

            robot_handle = self.gym.create_actor(
                env, robot_asset, pose, f"tron2_arm_{i}", i, 1
            )
            self.actor_handles.append(robot_handle)

            # Spawn target object
            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.4, 0.0, 0.05)
            obj_pose.r = gymapi.Quat(0, 0, 0, 1)

            obj_handle = self.gym.create_actor(
                env, obj_asset, obj_pose, f"target_object_{i}", i, 0
            )
            self.object_handles.append(obj_handle)

            self.envs.append(env)

        # Setup tensor access
        self._setup_tensors()

    def _setup_tensors(self):
        """Acquire GPU state tensors."""
        self.gym.prepare_sim(self.sim)

        # Robot state
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state = gymtorch.wrap_tensor(actor_root_state)
        self.root_state = self.root_state.view(self.num_envs, -1, 13)

        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.dof_state.view(self.num_envs, -1, 2)

        # Forces
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces)
        self.contact_forces = self.contact_forces.view(self.num_envs, -1, 3)

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        # Reset root state
        self.root_state[env_ids, 0, 0:3] = to_torch(
            self.cfg.init_state.pos, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)

        # Reset DOF positions
        num_dofs = self.gym.get_actor_dof_count(self.envs[0], self.actor_handles[0])
        for env_id in env_ids:
            for dof_idx, (joint_name, angle) in enumerate(
                self.cfg.init_state.default_joint_angles.items()
            ):
                if dof_idx < num_dofs:
                    self.dof_state[env_id, dof_idx, 0] = angle
                    self.dof_state[env_id, dof_idx, 1] = 0.0  # zero velocity

        # Reset object position with randomization
        for env_id in env_ids:
            obj_pos = torch_rand_float(
                [0.2, -0.15, 0.05],
                [0.5, 0.15, 0.15],
                (1, 3),
                device=self.device,
            ).squeeze()
            self.root_state[env_id, 1, 0:3] = obj_pos

        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

    def compute_observations(self):
        """Compute observations from current state."""
        raise NotImplementedError("Override in subclass with specific obs computation")

    def compute_reward(self):
        """Compute rewards from current state."""
        raise NotImplementedError("Override in subclass with specific reward function")

    def step(self, actions):
        """Execute one environment step."""
        # Apply actions
        actions = torch.clamp(actions, -self.cfg.normalization.clip_actions,
                              self.cfg.normalization.clip_actions)
        self.apply_actions(actions)

        # Step simulation
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Compute reward
        self.compute_reward()

        # Check termination
        self._check_termination()

        # Increment episode counter
        self.episode_length_buf += 1
        self.timeout_buf = self.episode_length_buf >= self.cfg.env.episode_length_s / self.cfg.sim.dt

        # Compute observations for next step
        obs = self.compute_observations()

        # Reset failed/successful episodes
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        return obs, self.rew_buf, self.reset_buf, self.timeout_buf

    def apply_actions(self, actions):
        """Apply actions to the robot DOFs."""
        num_dofs = self.gym.get_actor_dof_count(self.envs[0], self.actor_handles[0])

        actions = actions.view(self.num_envs, num_dofs)
        for env_idx in range(self.num_envs):
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_idx], self.actor_handles[env_idx])
            for dof_idx in range(num_dofs):
                target = actions[env_idx, dof_idx].item()
                self.gym.set_dof_target_position(
                    self.envs[env_idx],
                    self.actor_handles[env_idx],
                    dof_idx,
                    target,
                )

    def _check_termination(self):
        """Check for episode termination conditions."""
        # Reset on timeout
        self.reset_buf |= self.timeout_buf

        # Check for undesired contacts
        contact_forces = self.contact_forces[:, self.actor_handles, :]
        contact_norm = torch.norm(contact_forces, dim=-1)
        self.reset_buf |= torch.any(contact_norm > 500.0, dim=-1)
