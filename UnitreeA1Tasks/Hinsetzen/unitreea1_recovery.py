# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.unitreea1 import UnitreeA1
from omniisaacgymenvs.robots.articulations.views.unitreea1_view import UnitreeA1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math

from rl_games.torch_runner import Runner

class UnitreeA1RecoveryTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.init_done = False

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale_T = self._task_cfg["env"]["control"]["actionScaleT"]
        self.action_scale_P = self._task_cfg["env"]["control"]["actionScaleP"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"] 
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orientation"] = self._task_cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["deviation"] = self._task_cfg["env"]["learn"]["deviationRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["deviation"] = self._task_cfg["env"]["learn"]["deviationRewardScale"]
        self.rew_scales["jointSpeed"] = self._task_cfg["env"]["learn"]["jointSpeedRewardScale"]


        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.control_type = self._task_cfg["env"]["control"]["controlType"]
        self.base_threshold = 0.2
        self.knee_threshold = 0.1 # Noch nicht korrekt

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._unitreea1_translation = torch.tensor([0.0, 0.0, 1.0])
        self._unitreea1_orientation = torch.tensor([0.0, 0.0, -1.0, 0.0])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 69
        self._num_actions = 12
        
        self._a1_max_torque = 33.0
        self._a1_min_torque = self._a1_max_torque * (-1)

        self.x_coord = 0
        self.y_coord = 1
        self.z_coord = 2

        self.curriculum_factor = 0.3
        self.cf_exponent = 0.997

        self.current_state = "self_righting"

        self.common_step_counter = 0
        self.time_to_init = 69

        self.num_joints_per_leg = 3
        self.num_legs = 4

        RLTask.__init__(self, name, env)
        self.dof_limits = [[-0.8, -1.04, -2.7], [0.8, 4.18, -0.92]]
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_dof_pos = torch.tensor([0., 0., 0., 0., 2.6, 2.6, 2.6, 2.6, -2.7, -2.7, -2.7, -2.7], dtype=torch.float, device=self.device, requires_grad=False).repeat(self.num_envs,1)
        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat((self._num_envs, 1))
        self.forward_vec = torch.tensor([1., 0., 0.], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        return

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_unitreea1()
        super().set_up_scene(scene)
        self._unitrees = UnitreeA1View(prim_paths_expr="/World/envs/.*/A1", name="unitreea1view")
        scene.add(self._unitrees)
        scene.add(self._unitrees._feet)
        scene.add(self._unitrees._base)

        return

    def get_unitreea1(self):
        unitreea1 = UnitreeA1(prim_path=self.default_zero_env_path + "/A1", name="UnitreeA1Recovery", translation=self._unitreea1_translation, orientation = self._unitreea1_orientation)
        self._sim_config.apply_articulation_settings("UnitreeA1Recovery", get_prim_at_path(unitreea1.prim_path), self._sim_config.parse_actor_config("UnitreeA1Recovery"))
        #unitreea1.set_unitreea1_properties(self._stage, unitreea1.prim)
        
        #Configure joint properties
        if self.control_type == 'PD':
            joint_paths = ['FL_hip/FL_hip_fixed', 'FL_hip/FL_thigh_joint', 'FL_thigh/FL_calf_joint',
                            'FR_hip/FR_hip_fixed', 'FR_hip/FR_thigh_joint', 'FR_thigh/FR_calf_joint',
                            'RL_hip/RL_hip_fixed', 'RL_hip/RL_thigh_joint', 'RL_thigh/RL_calf_joint',
                            'RR_hip/RR_hip_fixed', 'RR_hip/RR_thigh_joint', 'RR_thigh/RR_calf_joint']
            for joint_path in joint_paths:
                set_drive(f"{unitreea1.prim_path}/{joint_path}", "angular", "position", 0, self.Kp, self.Kd, 33.0)

        dof_names = unitreea1.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._unitrees.get_world_poses()

        dof_limits = self._unitrees.get_dof_limits()
        self.unitree_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.unitree_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.num_dof = self._unitrees.num_dof
        self.current_targets = self.default_dof_pos.clone()
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)

        positions_offset = torch_rand_float(0.0, 0.0, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.0, 0.0, (len(env_ids), self.num_dof), device=self.device)

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        dof_pos = self.default_dof_pos[env_ids]
        for i in range(self.num_joints_per_leg):
            for j in range(i*self.num_legs, i*self.num_legs+self.num_legs):
                dof_pos[:, j] += torch_rand_float(self.dof_limits[0][i], self.dof_limits[1][i], (len(env_ids), 1), device=self.device).squeeze(-1)
        self.current_targets[env_ids] = dof_pos[:]

        self.dof_vel[env_ids] = velocities

        self._unitrees.set_world_poses(positions=self.initial_root_pos[env_ids].clone(), 
                                      orientations=self.initial_root_rot[env_ids].clone(),
                                      indices=indices)
        self._unitrees.set_velocities(velocities=root_vel,
                                          indices=indices)
        self._unitrees.set_joint_positions(positions=dof_pos[env_ids].clone(), 
                                          indices=indices)
        self._unitrees.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), 
                                          indices=indices)


        self.reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.

    def get_observations(self):
        self.obs_buf = torch.cat((  self.base_lin_vel * self.lin_vel_scale,
                                    self.base_ang_vel  * self.ang_vel_scale,
                                    self.projected_gravity,
                                    self.dof_pos * self.dof_pos_scale,
                                    self.last_dof_pos * self.dof_pos_scale,
                                    self.dof_vel * self.dof_vel_scale,
                                    self.last_dof_vel * self.dof_pos_scale,
                                    self.last_actions 
                                    ),dim=-1)

    def refresh_dof_state_tensors(self):
        self.dof_pos = self._unitrees.get_joint_positions(clone=False)
        self.dof_vel = self._unitrees.get_joint_velocities(clone=False)
    
    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._unitrees.get_world_poses(clone=False)
        self.base_velocities = self._unitrees.get_velocities(clone=False)

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return
        if (self.progress_buf < self.time_to_init).any():
            return
        
        self.actions = actions.clone().to(self.device)
        unready_envs = (self.progress_buf < self.time_to_init).nonzero(as_tuple=False).squeeze(-1)

        if self.control_type == "PD":
                indices = torch.arange(self._unitrees.count, dtype=torch.int32, device=self._device)
                self.actions[:] = actions.clone().to(self._device)
                current_targets = self.current_targets + self.action_scale_P * self.actions * self.dt 
                self.current_targets[:] = tensor_clamp(current_targets, self.unitree_dof_lower_limits, self.unitree_dof_upper_limits)
                self._unitrees.set_joint_position_targets(self.current_targets, indices)
        elif self.control_type == "T":
            torques = torch.clip(self.action_scale_T * self.actions * self.dt, -33.0, 33.0)
            torques[unready_envs] = torch.zeros((unready_envs.size(dim=0), self.num_actions), device=self.device, requires_grad=False)
            self._unitrees.set_joint_efforts(torques)
            self.torques = torques

    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            self.common_step_counter += 1
            if self.common_step_counter % 10000 == 0:
                self.curriculum_factor = self.curriculum_factor ** self.cf_exponent
            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
    
            feet_pos, _ = self._unitrees._feet.get_world_poses(clone=False)
            self.feet_heights = feet_pos.reshape(self.num_envs, 4, 3)[:, :, 2]
            feet_lin_vel = self._unitrees._feet.get_linear_velocities(clone=False)
            self.feet_xy_vel = feet_lin_vel.reshape(self.num_envs, 4, 3)[:, :, :2]

            self.check_termination()
            self.calculate_metrics()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()

            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]
            self.last_dof_pos[:] = self.dof_pos[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:3] = torch_rand_float(-1., 1., (self.num_envs, 3), device=self.device) # lin vel x/y
        self._unitrees.set_velocities(self.base_velocities)

    def calculate_metrics(self):
        # deviation from default joint configuration penalty
        rew_deviation = anglediff(self.goal_dof_pos, self.dof_pos).pow(2).sum(1).sqrt() * self.rew_scales["deviation"]

        # joint velocity penalty
        rew_joint_speed = torch.sum(torch.square(self.dof_vel), dim=1) * self.rew_scales["jointSpeed"] * self.curriculum_factor

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"] * self.curriculum_factor

        # torque penalty
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"] * self.curriculum_factor

        # orientation penalty
        rew_orientation = (self.gravity_vec - self.projected_gravity).pow(2).sum(1) * self.rew_scales["orientation"]

        # action rate penalty
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"] * self.curriculum_factor

        # total reward
        self.rew_buf = rew_orientation + \
                    rew_torque + rew_joint_acc + rew_action_rate + rew_deviation  + rew_joint_speed

        
        unready_envs = (self.progress_buf < self.time_to_init).nonzero(as_tuple=False).squeeze(-1)
        self.rew_buf[unready_envs] = 0.0

    def check_termination(self):
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        self.reset_buf = self.timeout_buf.clone()

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

def anglediff(target, source):
    minAngles = torch.abs(anglemod(target-source))
    return minAngles

def anglemod(a):
    return wrapAngle((a+math.pi)) - math.pi

def wrapAngle(a):
    twopi = 2*math.pi
    wrapAngleOutput = a - twopi * torch.floor(a/twopi)
    return wrapAngleOutput