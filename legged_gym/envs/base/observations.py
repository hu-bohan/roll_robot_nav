import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from legged_gym.envs.base.legged_robot import LeggedRobot as env1
else: # 运行的时候如果直接import会循环引用
    from legged_gym.envs.base.base_task import BaseTask as env1

def obs_base_lin_vel(env: env1, add_noise: bool):
    obs = env.base_lin_vel
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.lin_vel
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs * env.obs_scales.lin_vel

def obs_base_ang_vel(env: env1, add_noise: bool):
    obs = env.base_ang_vel
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.ang_vel
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs * env.obs_scales.ang_vel

def obs_projected_gravity(env: env1, add_noise: bool):
    obs = env.projected_gravity
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.gravity
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs

def obs_commands(env: env1):
    obs = env.commands[:, :3] * env.commands_scale
    # commands没有noise
    return obs

def obs_dof_pos(env: env1, add_noise: bool):
    obs = (env.dof_pos - env.default_dof_pos)

    if env.cfg.domain_rand.randomize_motor_offset:
        # 这里和compute_torque里的相反
        obs -= env.motor_offsets

    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.dof_pos
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs * env.obs_scales.dof_pos

def obs_dof_vel(env: env1, add_noise: bool):
    obs = env.dof_vel
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.dof_vel
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs * env.obs_scales.dof_vel

def obs_last_actions(env: env1):
    obs = env.actions
    # last action没有noise
    return obs

def obs_measure_heights(env: env1, add_noise: bool):
    heights = env.root_states[:, 2].unsqueeze(1) - 0.5 - env.measured_heights # 减去default height(0.5)归一化
    obs = torch.clip(heights, -1.0, 1.0)
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.height_measurements
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level
    return obs * env.obs_scales.height_measurements

