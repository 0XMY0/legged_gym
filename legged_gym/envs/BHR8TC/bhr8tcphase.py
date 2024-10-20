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

# Chz
# added variables:
# self.phase: Tensor num_envs, phase variable from observation
# self.dphase: Tensor num_envs, clipped action for phase change
# self.base_euler: Tensor num_envs * 3, euler angles of base orientation in (-pi, pi)
# self.footl_euler: Tensor num_envs * 3, euler angles of left foot orientation in (-pi, pi)
# self.footr_euler: Tensor num_envs * 3, euler angles of right foot orientation in (-pi, pi)
# self.rigid_state: Tensor num_envs * num_body * 13, data of the rigid bodies
# self.phase_swingl: Tensor num_envs, phase variable for left foot (1 for swing phase, 0 for stance phase)
# self.phase_swingr: Tensor num_envs, phase variable for right foot (1 for swing phase, 0 for stance phase)
# self.commandsy: Tensor num_envs, extra y velocity command with lipm
# self.commandsy_bound: Tensor num_envs, bound for self.commandsy, only change wrt height and desired phase

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi

class BHR8TCPHASE(LeggedRobot):
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.rigid_state = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dphase = torch.zeros(self.num_envs, device=self.device)
        self.base_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.footl_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.footr_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.phase_swingl = torch.zeros(self.num_envs, device=self.device)
        self.phase_swingr = torch.zeros(self.num_envs, device=self.device)
        self.commandsy = torch.zeros(self.num_envs, device=self.device)
        self.commandsy_bound = torch.zeros(self.num_envs, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.phase[env_ids] = 0

    def get_euler_xyz_clip(self, quat):
        euler = get_euler_xyz(quat)
        return torch.cat((
            wrap_to_pi(euler[0]).unsqueeze(1),
            wrap_to_pi(euler[1]).unsqueeze(1),
            wrap_to_pi(euler[2]).unsqueeze(1)
            ), dim=-1)
    
    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        dphase_bounds = self.cfg.control.dphase_bounds
        self.dphase = (self.actions[:, self.num_dof] + 1) / 2 * (dphase_bounds[1] - dphase_bounds[0]) + dphase_bounds[0]
        self.phase += self.dphase * self.sim_params.dt * self.cfg.control.decimation
        self.phase = torch.fmod(self.phase, 1)
        self.base_euler = self.get_euler_xyz_clip(self.base_quat)
        self.footl_euler = self.get_euler_xyz_clip(self.rigid_state[:, self.feet_indices[0], 3: 7])
        self.footr_euler = self.get_euler_xyz_clip(self.rigid_state[:, self.feet_indices[1], 3: 7])
        # get phase
        def _getphase(x, mu_A, mu_B):
            def normal_cdf(x, mu, sigma):
                return 0.5 * (1 + torch.erf((x - mu) / (sigma * 2.0)))
            P1 = normal_cdf(x, mu_A, 0.01) * (1 - normal_cdf(x, mu_B, 0.01))
            P2 = normal_cdf(x - 1, mu_A, 0.01) * (1 - normal_cdf(x - 1, mu_B, 0.01))
            P3 = normal_cdf(x + 1, mu_A, 0.01) * (1 - normal_cdf(x + 1, mu_B, 0.01))
            return P1 + P2 + P3
            # if(mu_A > 0.2 and mu_B < 0.8):
            #     return P1
            # if(mu_A < 0.2):
            #     P2 = normal_cdf(x - 1, mu_A, 0.01) * (1 - normal_cdf(x - 1, mu_B, 0.01))
            #     return P1 + P2
            # if(mu_B > 0.8):
            #     P3 = normal_cdf(x + 1, mu_A, 0.01) * (1 - normal_cdf(x + 1, mu_B, 0.01))
            #     return P1 + P3
            # return P1

        def _getphaseswinglr_walk():
            phase_swingl = _getphase(self.phase, 0.0, 0.45)
            phase_swingr = _getphase(self.phase, 0.5, 0.95)
            return phase_swingl, phase_swingr

        self.phase_swingl = _getphase(self.phase, 0.0, self.commands[:, 6])
        self.phase_swingr = _getphase(self.phase, self.commands[:, 5], self.commands[:, 5] + self.commands[:, 6])
        super()._post_physics_step_callback()

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        dphase_bounds = self.cfg.control.dphase_bounds
        self.commands[env_ids, 4] = torch_rand_float(dphase_bounds[0], dphase_bounds[1], (len(env_ids), 1), device=self.device).squeeze(1)
        x = self.cfg.rewards.base_height_target - 0.2
        y = self.commands[env_ids, 4]
        self.commandsy_bound[env_ids] = 0 # temp
        self.commands[:, 5] = torch.rand_like(self.commands[:, 5]) * 0.5
        self.commands[:, 6] = torch.rand_like(self.commands[:, 6]) * 0.5 + 0.3

    def compute_observations(self):
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.base_euler,
                                    self.commands[:, 0].unsqueeze(1) * self.commands_scale[0],
                                    self.commandsy.unsqueeze(1) * self.commands_scale[1],
                                    self.commands[:, 2].unsqueeze(1) * self.commands_scale[2],
                                    ((self.commands[:, 4] - self.cfg.control.dphase_bounds[0]) / (self.cfg.control.dphase_bounds[1] - self.cfg.control.dphase_bounds[0])).unsqueeze(1),
                                    self.commands[:, 5].unsqueeze(1),
                                    self.commands[:, 6].unsqueeze(1),
                                    (self.dof_pos - self.dof_pos_limits[:, 0]) / (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.phase.unsqueeze(1)
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
    
    def _compute_torques(self, actions):
        actions_scaled = actions[:, 0: self.num_dof] * self.cfg.control.action_scale # not used
        # cast from [-1, 1] to [-action_outscale, 1 + action_outscale]
        action_outscaled = (actions[:, 0: self.num_dof] + 1) / 2 * (1 + 2 * self.cfg.control.action_outscale) - self.cfg.control.action_outscale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            # casted to actual joint bounds
            action_outscaled_casted = action_outscaled * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) + self.dof_pos_limits[:, 0]
            torques = self.p_gains*(action_outscaled_casted - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _reset_dofs(self, env_ids):
        # reset with fixed positions
        self.dof_pos[env_ids] = ((self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1]) / 2).unsqueeze(0).expand(len(env_ids), -1)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def mirror_obs(self, obs):
        # Create the mirrored components
        mirrored_base_lin_vel = self.base_lin_vel.clone()
        mirrored_base_lin_vel[:, 1] = -mirrored_base_lin_vel[:, 1]
        mirrored_base_ang_vel = self.base_ang_vel.clone()
        mirrored_base_ang_vel[:, 0] = -mirrored_base_ang_vel[:, 0]
        mirrored_base_ang_vel[:, 2] = -mirrored_base_ang_vel[:, 2]
        mirrored_base_euler = self.base_euler.clone()
        mirrored_base_euler[:, 0] = -mirrored_base_euler[:, 0]
        mirrored_base_euler[:, 2] = -mirrored_base_euler[:, 2]
        mirrored_commands = self.commands.clone()
        mirrored_commands[:, 1] = self.commandsy.clone()
        mirrored_commands[:, 1: 3] = -mirrored_commands[:, 1: 3]
        mirrored_dof_pos = self.dof_pos.clone()
        mirrored_dof_pos[:, 0: 2] = -self.dof_pos[:, 5: 7]
        mirrored_dof_pos[:, 2: 5] = self.dof_pos[:, 7: 10]
        mirrored_dof_pos[:, 5: 7] = -self.dof_pos[:, 0: 2]
        mirrored_dof_pos[:, 7: 10] = self.dof_pos[:, 2: 5]
        mirrored_dof_vel = self.dof_vel.clone()
        mirrored_dof_vel[:, 0: 2] = -self.dof_vel[:, 5: 7]
        mirrored_dof_vel[:, 2: 5] = self.dof_vel[:, 7: 10]
        mirrored_dof_vel[:, 5: 7] = -self.dof_vel[:, 0: 2]
        mirrored_dof_vel[:, 7: 10] = self.dof_vel[:, 2: 5]
        mirrored_phase = (self.phase + 0.5) % 1

        mirrored_obs_buf = torch.cat((  mirrored_base_lin_vel * self.obs_scales.lin_vel,
                                        mirrored_base_ang_vel  * self.obs_scales.ang_vel,
                                        mirrored_base_euler,
                                        mirrored_commands[:, 0].unsqueeze(1) * self.commands_scale[0],
                                        mirrored_commands[:, 1].unsqueeze(1) * self.commands_scale[1],
                                        mirrored_commands[:, 2].unsqueeze(1) * self.commands_scale[2],
                                        ((mirrored_commands[:, 4] - self.cfg.control.dphase_bounds[0]) / (self.cfg.control.dphase_bounds[1] - self.cfg.control.dphase_bounds[0])).unsqueeze(1),
                                        mirrored_commands[:, 5].unsqueeze(1),
                                        mirrored_commands[:, 6].unsqueeze(1),
                                        (mirrored_dof_pos - self.dof_pos_limits[:, 0]) / (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]) * self.obs_scales.dof_pos,
                                        mirrored_dof_vel * self.obs_scales.dof_vel,
                                        mirrored_phase.unsqueeze(1)
                                        ),dim=-1)
        return mirrored_obs_buf
    
    def mirror_actions(self, actions):
        mirrored_actions = actions.clone()
        mirrored_actions[:, 0: 2] = -actions[:, 5: 7]
        mirrored_actions[:, 2: 5] = actions[:, 7: 10]
        mirrored_actions[:, 5: 7] = -actions[:, 0: 2]
        mirrored_actions[:, 7: 10] = actions[:, 2: 5]
        return mirrored_actions

    def _reward_tracking_lin_vel(self):
        # use a linear function of the bound to approximate desired, and clip the desired vy for the first timesteps
        episode_time = self.episode_length_buf * self.cfg.control.decimation * self.sim_params.dt
        commandy_scale = torch.where(episode_time * self.commands[:, 4] < 0.5, 0.0, 
            torch.where(episode_time * self.commands[:, 4] > 1, 1.0, 
            episode_time * self.commands[:, 4] * 2 - 1))    
        self.commandsy = commandy_scale * (torch.where(self.phase < 0.5, -self.commandsy_bound + (self.phase - 0) * 4 * self.commandsy_bound, self.commandsy_bound - (self.phase - 0.5) * 4 * self.commandsy_bound)) + self.commands[:, 1]
        #lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]) + torch.square(self.commandsy - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_phase_regulation_force(self):
        forcel = torch.norm(self.contact_forces[:, self.feet_indices[0], :], dim=1)
        forcer = torch.norm(self.contact_forces[:, self.feet_indices[1], :], dim=1)
        return self.phase_swingl * forcel + self.phase_swingr * forcer

    def _reward_phase_regulation_vel(self):
        vl = torch.norm(self.rigid_state[:, self.feet_indices[0], 7: 9], dim=1)
        vr = torch.norm(self.rigid_state[:, self.feet_indices[1], 7: 9], dim=1)
        return (1.0 - self.phase_swingl) * vl + (1.0 - self.phase_swingr) * vr
        
    def _reward_foot_clearance(self):
        # Penalize foot height difference from desired
        zl = self.rigid_state[:, self.feet_indices[0], 2]
        zr = self.rigid_state[:, self.feet_indices[1], 2]
        zmax = 0.02 + 0.1 * self.commands[:, 6]
        phasel = torch.clip((self.phase - 0.0) / (self.commands[:, 6] - 0.0), 0, 1)
        phaser = torch.where(
            self.phase < self.commands[:, 5], 
            torch.clip((self.phase + 1 - self.commands[:, 5]) / (self.commands[:, 6] - self.commands[:, 5]), 0, 1),
            torch.clip((self.phase - self.commands[:, 5]) / (self.commands[:, 6] - self.commands[:, 5]), 0, 1)
            )
        zl_desired = zmax * torch.sin(phasel * np.pi)
        zr_desired = zmax * torch.sin(phaser * np.pi)
        return torch.square((zl - zr) - (zl_desired - zr_desired))

    def _reward_feet_distancey(self):
        # Penalize feet distance in y direction
        pl_w = self.rigid_state[:, self.feet_indices[0], 0: 3]
        pr_w = self.rigid_state[:, self.feet_indices[1], 0: 3]
        pbody = self.rigid_state[:, 0, 0: 3]
        q_conj = torch.cat((-self.base_quat[:, :3], self.base_quat[:, 3].unsqueeze(1)), dim=-1)
        pl_b = quat_apply_yaw(q_conj, pl_w - pbody)
        pr_b = quat_apply_yaw(q_conj, pr_w - pbody)
        return torch.exp(-30.0 * torch.square(pl_b[:, 1] - pr_b[:, 1] - 0.2))
    
    def _reward_feet_orientation(self):
        return torch.sum(torch.square(self.footl_euler[:, :2]), dim=1) + torch.sum(torch.square(self.footr_euler[:, :2]), dim=1)
    
    def _reward_feet_yaw_wrt_base(self):
        return torch.square(wrap_to_pi(self.footl_euler[:, 2] - self.base_euler[:, 2])) + torch.square(wrap_to_pi(self.footr_euler[:, 2] - self.base_euler[:, 2]))

    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        if_contact = torch.sum(1.*contacts, dim=1) >= 1
        return 1.*if_contact
    
    def _reward_tracking_dphase(self):
        return torch.square(self.dphase - self.commands[:, 4])
    
    def _reward_base_height_wrt_foot(self):
        base_height = self.root_states[:, 2]
        min_foot_height, _ = torch.min(self.rigid_state[:, self.feet_indices, 2], dim=1)
        base_height_wrt_foot = base_height - min_foot_height
        return torch.square(base_height_wrt_foot - self.cfg.rewards.base_height_wrt_foot_target)
    
    def _reward_stay_alive(self):
        return self.episode_length_buf < 100
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

        # check if the robot is too low
        uppbody_too_low_buf = (self.root_states[:, 2]) < 0.2
        self.reset_buf |= uppbody_too_low_buf
        
        # check if the robot posture is too bad
        roll_outbound_buf = (self.base_euler[:, 0] < -0.8) | (self.base_euler[:, 0] > 0.8)
        pitch_outbound_buf = (self.base_euler[:, 1] < -0.8) | (self.base_euler[:, 1] > 0.8)
        self.reset_buf |= (roll_outbound_buf | pitch_outbound_buf)

        # check if the feet are too close
        pl_w = self.rigid_state[:, self.feet_indices[0], 0: 3]
        pr_w = self.rigid_state[:, self.feet_indices[1], 0: 3]
        pbody = self.rigid_state[:, 0, 0: 3]
        q_conj = torch.cat((-self.base_quat[:, :3], self.base_quat[:, 3].unsqueeze(1)), dim=-1)
        pl_b = quat_apply_yaw(q_conj, pl_w - pbody)
        pr_b = quat_apply_yaw(q_conj, pr_w - pbody)
        feet_dis_buf = (pl_b[:, 1] - pr_b[:, 1] < 0.07)
        self.reset_buf |= feet_dis_buf

        #check if the grf is too large
        # grf_buf = (self.contact_forces[:, self.feet_indices[0], 2] > 1500.0) | (self.contact_forces[:, self.feet_indices[1], 2] > 1500.0)
        # # not panelizing in the first 0.5s
        # grf_buf &= (self.episode_length_buf > 1.0 / (self.sim_params.dt * self.cfg.control.decimation))
        # self.reset_buf |= grf_buf