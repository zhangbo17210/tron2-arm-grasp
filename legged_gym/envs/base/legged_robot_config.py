"""
Base configuration classes for Tron2 Arm tasks.
Defines observation, action, and reward configuration structures.
"""

import torch


class Tron2ArmBaseConfig:
    """Mixin providing common PPO config fields."""

    class runner:
        max_iterations = 50000
        experiment_name = "tron2_arm_grasp"
        save_interval = 1000
        resume = False
        load_run = -1
        checkpoint = -1

    class algorithm:
        entropy_coef = 0.005
        learning_rate = 3e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.0
        max_grad_norm = 1.0
        desired_kl = 0.01
        schedule = "adaptive"

    class policy:
        init_noise_std = 1.0

        class actor:
            hidden_dims = [256, 128, 64]
            activation = "elu"

        class critic:
            hidden_dims = [256, 128, 64]
            activation = "elu"


class Tron2ArmCfg:
    """Base configuration for Tron2 arm grasping environment."""

    class env:
        num_envs = 4096
        num_observations = 48
        num_privileged_obs = num_observations + 12
        num_actions = 8
        env_spacing = 2.0
        send_timeouts = True
        episode_length_s = 8.0
        obs_history_length = 3

    class terrain:
        mesh_type = "trimesh"
        horizontal_scale = 0.05
        vertical_scale = 0.005
        border_size = 5.0
        static_friction = 0.8
        dynamic_friction = 0.6
        restitution = 0.0

    class init_state:
        pos = [0.0, 0.0, 0.4]
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        default_joint_angles = {
            "joint1": 0.0,
            "joint2": -0.3,
            "joint3": 0.0,
            "joint4": -1.0,
            "joint5": 0.0,
            "joint6": 0.8,
            "gripper_left": 0.0,
            "gripper_right": 0.0,
        }

    class control:
        control_type = "P"
        stiffness = {
            "joint1": 100.0,
            "joint2": 150.0,
            "joint3": 100.0,
            "joint4": 80.0,
            "joint5": 40.0,
            "joint6": 40.0,
            "gripper_left": 50.0,
            "gripper_right": 50.0,
        }
        damping = {
            "joint1": 2.0,
            "joint2": 3.0,
            "joint3": 2.0,
            "joint4": 1.5,
            "joint5": 1.0,
            "joint6": 1.0,
            "gripper_left": 1.0,
            "gripper_right": 1.0,
        }
        decimation = 4
        user_torque_limit = 50.0

    class commands:
        curriculum = True
        num_commands = 4
        resampling_time = 3.0

        class ranges:
            lin_pos_x = [0.2, 0.6]
            lin_pos_y = [-0.3, 0.3]
            lin_pos_z = [0.1, 0.5]

    class domain_rand:
        randomize_dof_pos = True
        randomize_dof_vel = True
        randomize_gripper = True
        randomize_object_pos = True
        randomize_object_size = True
        randomize_object_friction = True
        push_robots = False

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/tron2_arm/tron2_arm.urdf"
        name = "tron2_arm"
        penalize_contacts_on = ["base", "link1", "link2"]
        terminate_after_contacts_on = ["gripper_tip"]
        collapse_fixed_joints = True
        fix_base_link = True
        default_dof_drive_mode = 3
        self_collisions = 1

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 100.0
        max_linear_velocity = 1.0
        armature = 0.01
        thickness = 0.01

    class rewards:
        only_positive_rewards = True
        tracking_sigma = 0.01
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9

        class scales:
            reach_target = 2.0
            grasp_success = 5.0
            lift_object = 3.0
            action_rate = -0.005
            dof_pos_limits = -1.0
            dof_vel_limits = -0.5
            torque_limits = -0.01
            collision = -0.5
            orientation = -0.5

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            dof_acc = 0.0025
            gravity = 2.0

        clip_observations = 100.0
        clip_actions = 100.0

    class sim:
        dt = 0.005
        substeps = 2
        gravity = [0.0, 0.0, -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class Tron2ArmCfgPPO(Tron2ArmBaseConfig):
    """PPO algorithm configuration for Tron2 arm grasping."""

    class runner(Tron2ArmBaseConfig.runner):
        max_iterations = 50000
        experiment_name = "tron2_arm_grasp"
        save_interval = 1000
        resume = False

    class algorithm(Tron2ArmBaseConfig.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.0
        max_grad_norm = 1.0

    class policy(Tron2ArmBaseConfig.policy):
        init_noise_std = 1.0

        class actor(Tron2ArmBaseConfig.policy.actor):
            hidden_dims = [256, 128, 64]
            activation = "elu"

        class critic(Tron2ArmBaseConfig.policy.critic):
            hidden_dims = [256, 128, 64]
            activation = "elu"
