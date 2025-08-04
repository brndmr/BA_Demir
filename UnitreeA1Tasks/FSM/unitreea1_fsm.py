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

import onnx
import onnxruntime as ort

class UnitreeA1FSMTask(RLTask):
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

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._unitreea1_translation = torch.tensor([0.0, 0.0, 1.0])
        self._unitreea1_orientation = torch.tensor([0.0, 0.0, -1.0, 0.0])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 70
        self._num_actions = 12
        
        self._a1_max_torque = 33.0
        self._a1_min_torque = self._a1_max_torque * (-1)

        self.x_coord = 0
        self.y_coord = 1
        self.z_coord = 2

        self.recovery_state = 0
        self.standup_state = 1

        self.time_to_init = 200

        recovery_model = onnx.load("runs/UnitreeA1Recovery/nn/UnitreeA1Recovery.onnx")
        onnx.checker.check_model(recovery_model)
        self.recovery_model = ort.InferenceSession("runs/UnitreeA1Recovery/nn/UnitreeA1Recovery.onnx")
        outputs_recovery = self.recovery_model.run(
        None,
        {"obs": np.zeros((1, 69)).astype(np.float32)},
        )

        standing_up_model = onnx.load("runs/UnitreeA1StandUp/nn/UnitreeA1StandUp.onnx")
        onnx.checker.check_model(standing_up_model)
        self.standing_up_model = ort.InferenceSession("runs/UnitreeA1StandUp/nn/UnitreeA1StandUp.onnx")
        outputs_standing_up = self.standing_up_model.run(
        None,
        {"obs": np.zeros((1, 70)).astype(np.float32)},
        )

        self.action_scale = 10000

        RLTask.__init__(self, name, env)
        self.current_state = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device, requires_grad=False)
        self.dof_limits = [[-0.8, -1.04, -2.7], [0.8, 4.18, -0.92]]
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_dof_pos = torch.tensor([0., 0., 0., 0., 2.6, 2.6, 2.6, 2.6, -2.7, -2.7, -2.7, -2.7], dtype=torch.float, device=self.device, requires_grad=False)
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
        unitreea1 = UnitreeA1(prim_path=self.default_zero_env_path + "/A1", name="UnitreeA1FSM", translation=self._unitreea1_translation, orientation = self._unitreea1_orientation)
        self._sim_config.apply_articulation_settings("UnitreeA1FSM", get_prim_at_path(unitreea1.prim_path), self._sim_config.parse_actor_config("UnitreeA1FSM"))
        #unitreea1.set_unitreea1_properties(self._stage, unitreea1.prim)

        dof_names = unitreea1.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._unitrees.get_world_poses()

        self.num_dof = self._unitrees.num_dof
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_xyz = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):

        indices = env_ids.to(dtype=torch.int32)
        num_resets = len(env_ids)
        
        self.current_state[env_ids] = torch.zeros(num_resets, dtype=torch.uint8, device=self.device, requires_grad=False)
        
        positions_offset = torch_rand_float(0.75, 1.25, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.0, 0.0, (len(env_ids), self.num_dof), device=self.device)

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        dof_pos = self.default_dof_pos[env_ids]
        for i in range(3):
            for j in range(i*4, i*4+4):
                dof_pos[:, j] += torch_rand_float(self.dof_limits[0][i], self.dof_limits[1][i], (len(env_ids), 1), device=self.device).squeeze(-1)

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
        self.obs_buf = torch.cat((  self.base_pos[:, 2, None],
                                    self.base_lin_vel * self.lin_vel_scale,
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
        self.xyz_from_quat_tensor(self.base_quat)

        #print("BASE XYZ:", self.base_xyz)

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return
        
        if (self.progress_buf < self.time_to_init).any():
            return
        
        recovery_envs = (self.current_state == 0).nonzero(as_tuple=False).squeeze(-1).detach().cpu().numpy()
        standup_envs = (self.current_state == 1).nonzero(as_tuple=False).squeeze(-1).detach().cpu().numpy()

        recovery_obs = self.obs_buf[:,1:]
        outputs_recovery = self.recovery_model.run(None, {"obs" : recovery_obs.detach().cpu().numpy().astype(np.float32)})
        mu_recovery = outputs_recovery[0][recovery_envs]
        if recovery_envs.size > 0:
            self.actions[recovery_envs, :] = torch.clamp(torch.from_numpy(mu_recovery), -1.0, 1.0).clone().to(self.device)

        standup_obs = self.obs_buf[:, :]
        outputs_standup = self.standing_up_model.run(None, {"obs" : standup_obs.detach().cpu().numpy().astype(np.float32)})
        mu_standup = outputs_standup[0][standup_envs]
        if standup_envs.size > 0:
            self.actions[standup_envs, :] = torch.clamp(torch.from_numpy(mu_standup), -1.0, 1.0).clone().to(self.device)

        torques = torch.clip(self.action_scale * self.actions * self.dt, -33.0, 33.0)
        self._unitrees.set_joint_efforts(torques)
        self.torques = torques


    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])

            recovery_envs = torch.where(((self.current_state == self.recovery_state) & (self.base_xyz[:,0].abs() >= 0.2)) | ((self.current_state == self.standup_state) & (self.base_xyz[:,0].abs() >= 1.0)), 1, 0).nonzero(as_tuple=False).squeeze(-1)
            standup_envs = torch.where(((self.current_state == self.standup_state) & (self.base_xyz[:,0].abs() < 1.0)) | ((self.current_state == self.recovery_state) & (self.base_xyz[:,0].abs() < 0.2)), 1, 0).nonzero(as_tuple=False).squeeze(-1)

            self.current_state[recovery_envs] = self.recovery_state
            self.current_state[standup_envs] = self.standup_state

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

    def calculate_metrics(self):
        self.rew_buf[:] = 0.0

    def check_termination(self):
        self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        self.reset_buf = self.timeout_buf.clone()

    def xyz_from_quat_tensor(self, quat_tensor):
        """Convert a quaternion to XYZ euler angles.

        Args:
            quat (np.ndarray): A 4x1 vector in order (w, x, y, z).

        Returns:
            np.ndarray: A 3x1 vector containing (roll, pitch, yaw).
        """
        w = quat_tensor[:, 0].detach()
        x = quat_tensor[:, 1].detach()
        y = quat_tensor[:, 2].detach()
        z = quat_tensor[:, 3].detach()
        y_sqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y_sqr)
        eulerx = torch.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2[t2 > +1.0] = +1.0
        t2[t2 < -1.0] = -1.0
        eulery = torch.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y_sqr + z * z)
        eulerz = torch.arctan2(t3, t4)

        self.base_xyz[:, 0] = eulerx
        self.base_xyz[:, 1] = eulery
        self.base_xyz[:, 2] = eulerz

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