"""
Tron2 Arm Grasping Task Configuration
Task-specific hyperparameters for Isaac Gym training.
"""

from configs.tron2_arm_config import Tron2ArmCfg, Tron2ArmCfgPPO


class Tron2ArmGraspCfg(Tron2ArmCfg):
    """Task configuration for Tron2 arm grasping with cube objects."""

    class env(Tron2ArmCfg.env):
        num_observations = 52  # joint_pos(8) + joint_vel(8) + obj_pos(3) + obj_quat(4) + eef_pos(3) + cmd(4) + gravity(3) + prev_action(8) + height(1) = 37, pad to 52
        num_privileged_obs = num_observations + 8  # with contact forces
        num_actions = 8
        episode_length_s = 10.0

    class terrain(Tron2ArmCfg.terrain):
        mesh_type = "none"  # flat table surface
        static_friction = 1.0
        dynamic_friction = 0.8

    class init_state(Tron2ArmCfg.init_state):
        pos = [0.0, 0.0, 0.6]  # mounted on table
        default_joint_angles = {
            "joint1": 0.0,
            "joint2": -0.4,
            "joint3": 0.0,
            "joint4": -1.2,
            "joint5": 0.0,
            "joint6": 0.6,
            "gripper_left": 0.02,
            "gripper_right": -0.02,
        }

    class commands(Tron2ArmCfg.commands):
        num_commands = 3  # [target_x, target_y, target_z] for end-effector
        resampling_time = 4.0

        class ranges(Tron2ArmCfg.commands.ranges):
            lin_pos_x = [0.15, 0.55]
            lin_pos_y = [-0.3, 0.3]
            lin_pos_z = [0.05, 0.45]

    class rewards(Tron2ArmCfg.rewards):
        class scales(Tron2ArmCfg.rewards.scales):
            reach_target = 3.0
            grasp_success = 10.0
            lift_object = 5.0
            action_rate = -0.002
            dof_pos_limits = -2.0
            dof_vel_limits = -0.1
            torque_limits = -0.02
            collision = -1.0
            orientation = -0.3

    class domain_rand(Tron2ArmCfg.domain_rand):
        randomize_dof_pos = True
        randomize_dof_vel = True
        randomize_gripper = True
        randomize_object_pos = True
        randomize_object_size = True
        randomize_object_friction = True
        push_robots = False

    class asset(Tron2ArmCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/tron2_arm.urdf"

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


class Tron2ArmGraspCfgPPO(Tron2ArmCfgPPO):
    """PPO configuration for Tron2 arm grasping."""

    class runner(Tron2ArmCfgPPO.runner):
        max_iterations = 50000
        experiment_name = "tron2_arm_grasp_v1"
        save_interval = 1000
        resume = False

    class algorithm(Tron2ArmCfgPPO.algorithm):
        entropy_coef = 0.005
        learning_rate = 3e-4
        num_learning_epochs = 5
        num_mini_batches = 4
        clip_param = 0.2
        gamma = 0.99
        lam = 0.95
        value_loss_coef = 1.0
        max_grad_norm = 1.0
        schedule = "adaptive"
        desired_kl = 0.01

    class policy(Tron2ArmCfgPPO.policy):
        init_noise_std = 0.5

        class actor(Tron2ArmCfgPPO.policy.actor):
            hidden_dims = [512, 256, 128]
            activation = "elu"

        class critic(Tron2ArmCfgPPO.policy.critic):
            hidden_dims = [512, 256, 128]
            activation = "elu"
