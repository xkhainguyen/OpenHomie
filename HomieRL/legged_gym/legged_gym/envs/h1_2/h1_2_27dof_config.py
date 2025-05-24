# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class H12RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ): #TODO: change for h12
        pos = [0.0, 0.0, 1.05] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'left_hip_yaw_joint' : 0. ,   
            'left_hip_roll_joint' : 0,               
            'left_hip_pitch_joint' : -0.4,         
            'left_knee_joint' : 0.8,       
            'left_ankle_pitch_joint' : -0.4,     
            'left_ankle_roll_joint' : 0,     
            'right_hip_yaw_joint' : 0., 
            'right_hip_roll_joint' : 0, 
            'right_hip_pitch_joint' : -0.4,                                       
            'right_knee_joint' : 0.8,                                             
            'right_ankle_pitch_joint': -0.4,                              
            'right_ankle_roll_joint' : 0,         
            "torso_joint": 0.,
            "left_shoulder_pitch_joint": 0.4,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_yaw_joint": 0.,
            "left_elbow_joint": 0.3,
            "left_wrist_roll_joint": 0.,
            "left_wrist_pitch_joint": 0.,
            "left_wrist_yaw_joint": 0.,
            "right_shoulder_pitch_joint": 0.4,
            "right_shoulder_roll_joint": -0.2,#-0.3
            "right_shoulder_yaw_joint": 0.,
            "right_elbow_joint": 0.3,#0.8
            "right_wrist_roll_joint": 0.,
            "right_wrist_pitch_joint": 0.,
            "right_wrist_yaw_joint": 0.,
        }

    class control( LeggedRobotCfg.control ): #TODO: change for h12
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     "torso": 300,
                     "shoulder": 120,
                     "elbow": 80,
                     "wrist": 40,                     
                     }  # [N*m/rad]
        # damping = {  'hip_yaw': 2.5,
        #              'hip_roll': 2.5,
        #              'hip_pitch': 2.5,
        #              'knee': 4,
        #              'ankle': 2,
        #              "torso": 5,
        #              "shoulder": 4,
        #              "elbow": 1,
        #              "wrist": 0.5,
        #              }  # [N*m/rad]  # [N*m*s/rad]
        damping = {  'hip_yaw': 2.5,
                     'hip_roll': 2.5,
                     'hip_pitch': 2.5,
                     'knee': 4.0,
                     'ankle': 2.0,
                     "torso": 4.0,
                     "shoulder": 3.0,
                     "elbow": 2.0,
                     "wrist": 2.0,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = True # NOTE set True later
        max_curriculum = 1.4
        num_commands = 5 # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, height, orientation
        resampling_time = 8. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        heading_to_ang_vel = False
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-0.7, 0.7] # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]
            heading = [-3.14, 3.14]
            height = [-0.4, 0.0]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2_description/h1_2.urdf'
        name = "h1_2"
        foot_name = "ankle_roll"
        left_foot_name = "left_foot"
        right_foot_name = "right_foot"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ['torso']
        curriculum_joints = []
        left_leg_joints = ['left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_pitch_joint']
        right_leg_joints = ['right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_pitch_joint']
        left_hip_joints = ['left_hip_roll_joint', "left_hip_pitch_joint", "left_hip_yaw_joint"]
        right_hip_joints = ['right_hip_roll_joint', "right_hip_pitch_joint", "right_hip_yaw_joint"]
        hip_pitch_joints = ['right_hip_pitch_joint', 'left_hip_pitch_joint']
        knee_joints = ['left_knee_joint', 'right_knee_joint']
        ankle_joints = ["left_ankle_roll_joint", "right_ankle_roll_joint"]
        upper_body_link = "torso_link"
        imu_link = "imu"
        knee_names = ["left_knee_link", "left_hip_yaw_link", "right_knee_link", "right_hip_yaw_link"]
        hand_names = ["L_hand_base_link", "R_hand_base_link"]
        self_collision = 0
        flip_visual_attachments = False
        ankle_sole_distance = 0.04
        armature = 1e-3

        
    class domain_rand(LeggedRobotCfg.domain_rand):
        
        use_random = True
        
        randomize_joint_injection = use_random
        joint_injection_range = [-0.05, 0.05]
        
        randomize_actuation_offset = use_random
        actuation_offset_range = [-0.05, 0.05]

        randomize_payload_mass = use_random
        payload_mass_range = [-5, 10]
        
        hand_payload_mass_range = [-0.1, 0.3]

        randomize_com_displacement = False
        com_displacement_range = [-0.1, 0.1]
        
        randomize_body_displacement = use_random
        body_displacement_range = [-0.1, 0.1]

        randomize_link_mass = use_random
        link_mass_range = [0.9, 1.1]
        
        randomize_friction = use_random
        friction_range = [0.1, 1.25]
        
        randomize_restitution = use_random
        restitution_range = [0.0, 1.0]
        
        randomize_kp = use_random
        kp_range = [0.9, 1.1]
        
        randomize_kd = use_random
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = use_random
        initial_joint_pos_scale = [0.7, 1.3]
        initial_joint_pos_offset = [-0.2, 0.2]
        
        push_robots = use_random
        push_interval_s = 4
        upper_interval_s = 1
        max_push_vel_xy = 0.6
        
        init_upper_ratio = 0.
        delay = use_random

    class rewards( LeggedRobotCfg.rewards ): #TODO: change for h12
        class scales:
            tracking_x_vel = 1.5
            tracking_y_vel = 1.
            tracking_ang_vel = 2.
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            orientation = -1.5
            action_rate = -0.05
            tracking_base_height = 2.
            deviation_hip_joint = -0.2
            deviation_ankle_joint = -0.5
            deviation_knee_joint = -0.75
            dof_acc = -2.5e-7
            dof_pos_limits = -5.
            # feet_swing_height = -20.0
            # feet_air_time = 0.05
            feet_clearance = -0.25
            feet_distance_lateral = 0.5
            knee_distance_lateral = 1.0
            feet_ground_parallel = -2.0
            feet_parallel = -3.0
            smoothness = -0.1
            joint_power = -2e-5
            feet_stumble = -1.5
            torques = -2.5e-6
            dof_vel = -1e-4
            alive = 0.15
            # hip_pos = -1.0
            # contact_no_vel = -0.2
            dof_vel_limits = -2e-3
            torque_limits = -0.1
            no_fly = 0.75
            joint_tracking_error = -0.1
            feet_slip = -0.25
            feet_contact_forces = -0.00025
            contact_momentum = 2.5e-4
            action_vanish = -1.0
            stand_still = -0.15
            # stand_still_angle = -0.1   
        only_positive_rewards = True
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.80
        soft_torque_limit = 0.95
        base_height_target = 0.95
        max_contact_force = 700.
        least_feet_distance = 0.25
        # feet_swing_height_threshold = 0.08
        least_feet_distance_lateral = 0.25
        most_feet_distance_lateral = 0.45
        most_knee_distance_lateral = 0.45
        least_knee_distance_lateral = 0.2
        clearance_height_target = 0.14
        
    class env( LeggedRobotCfg.rewards ):
        num_envs = 4096
        num_actions = 12
        num_dofs = 27
        num_one_step_observations = 2 * num_dofs + 10 + num_actions # 54 + 10 + 12 = 22 + 54 = 76
        num_one_step_privileged_obs = num_one_step_observations + 3
        num_actor_history = 6
        num_critic_history = 1
        num_observations = num_actor_history * num_one_step_observations
        num_privileged_obs = num_critic_history * num_one_step_privileged_obs
        action_curriculum = True
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'

    class noise( LeggedRobotCfg.terrain ):
        add_noise = True
        noise_level = 1.0
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.02
            dof_vel = 2.0
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurement = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025

class H12RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.1
        actor_hidden_dims = [512, 256, 256]
        critic_hidden_dims = [512, 256, 256]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        use_flip = True
        entropy_coef = 0.01
        symmetry_scale = 1.0
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'HIMActorCritic'
        algorithm_class_name = 'HIMPPO'
        save_interval = 200
        num_steps_per_env = 20
        max_iterations = 100000
        run_name = ''
        experiment_name = ''
        wandb_project = ""
        # logger = "wandb"        
        logger = "tensorboard"        
        # wandb_user = "" # enter your own wandb user name here

# # ALMI config
# class H12RoughCfgPPO(BaseConfig):
#     seed = 1
#     runner_class_name = 'OnPolicyRunner'
#     class policy:
#         init_noise_std = 0.1
#         actor_hidden_dims = [512, 256, 256]
#         critic_hidden_dims = [512, 256, 256]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         # rnn_type = 'lstm'
#         # rnn_hidden_size = 512
#         # rnn_num_layers = 1
        
#     class algorithm:
#         # training params
#         value_loss_coef = 1.0
#         use_clipped_value_loss = True
#         clip_param = 0.2
#         entropy_coef = 0.01
#         num_learning_epochs = 5
#         num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
#         learning_rate = 1.e-3 #5.e-4
#         schedule = 'adaptive' # could be adaptive, fixed
#         gamma = 0.99
#         lam = 0.95
#         desired_kl = 0.01
#         max_grad_norm = 1.

#     class runner:
#         policy_class_name = 'ActorCritic'
#         algorithm_class_name = 'PPO'
#         num_steps_per_env = 24 # per iteration
#         max_iterations = 1500 # number of policy updates

#         # logging
#         save_interval = 200
#         num_steps_per_env = 50
#         max_iterations = 100000
#         run_name = ''
#         experiment_name = ''
#         wandb_project = ""
#         # logger = "wandb"        
#         logger = "tensorboard"      