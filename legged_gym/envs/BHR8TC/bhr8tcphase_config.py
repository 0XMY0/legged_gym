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

class BHR8TCPHASERoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 1024
        num_dofs = 10
        num_actions = sum([
            num_dofs,       # dof_target_vel
            1               # delta_phase
            ])
        num_observations = sum([
            3,              # base lin vel
            3,              # base ang vel
            3,              # projected_gravity
            3,              # commands
            num_dofs,       # dof_pos
            num_dofs,       # dof_vel
            1               # phase variable
            ])
        episode_length_s = 20.0
    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands( LeggedRobotCfg.commands ):
        resampling_time = 10.0

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 0.87] # x,y,z,w [quat]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'leg1_left': 0.0,
            'leg2_left': 0.0,
            'leg3_left': -0.3,
            'leg4_left': 0.6,
            'leg5_left': -0.3,

            'leg1_right': 0.0,
            'leg2_right': 0.0,
            'leg3_right': -0.3,
            'leg4_right': 0.6,
            'leg5_right': -0.3
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {
            'leg1': 100.0,
            'leg2': 100.0,
            'leg3': 200.0,
            'leg4': 200.0,
            'leg5': 200.0}  # [N*m/rad]
        damping = {
            'leg1': 3.0,
            'leg2': 3.0,
            'leg3': 6.0,
            'leg4': 6.0,
            'leg5': 6.0}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 20
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/BHR8TC/bhr8tc.urdf'
        name = "bhr8tcphase"
        foot_name = 'ankle'
        terminate_after_contacts_on = ['uppbody']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand ):
        push_interval_s = 7.0
        randomize_base_mass = True
        # added_mass_range = [-3., 3.]
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 600.
        # base_height_target = 1.15
        base_height_wrt_foot_target = 0.75
        base_height_target = 0.75
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.5
            torques = -3.e-6
            dof_acc = -2.e-7
            action_rate = -0.01
            lin_vel_z = -0.5
            feet_air_time = 0.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            orientation = -5.0
            feet_contact_forces = -5.e-3
            tracking_dphase = -1.0
            phase_regulation_force = -5.e-3
            phase_regulation_vel = -1.0
            foot_clearance = -10.0
            feet_distancey = 1.0
            feet_orientation = -0.3
            feet_yaw_wrt_base = -3.0
            base_height_wrt_foot = -0.0
            base_height = -10.0
            stay_alive = 0.1

    class sim( LeggedRobotCfg.sim ):
        dt =  0.001

class BHR8TCPHASERoughCfgPPO( LeggedRobotCfgPPO ):

    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 64
        # rnn_num_layers = 2

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCritic' # 'ActorCritic'
        run_name = ''
        experiment_name = 'rough_bhr8tcphase'
        num_steps_per_env = 48

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3
        num_learning_epochs = 5
        num_mini_batches = 8
        



  