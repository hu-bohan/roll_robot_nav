import torch
from legged_gym.envs.base.observations import * # 把base环境的reward函数import过来

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from legged_gym.envs.roll_robot_r.roll_robot_r_amp import rollRobotRAMP as env1
else: # 运行的时候如果直接import会循环引用
    from legged_gym.envs.base.base_task import BaseTask as env1

def obs_dof_pos_normalized(env: env1, add_noise: bool):
    """在关节limit内归一化
    零点居中的对称映射
    """
    obs = (env.dof_pos - env.default_dof_pos)

    if env.cfg.domain_rand.randomize_motor_offset:
        # 这里和compute_torque里的相反
        obs -= env.motor_offsets

    # 在归一化之前添加dof pos noise
    if add_noise:
        global_noise_level = env.get_global_noise_level()
        noise_scale = env.cfg.noise.noise_scales.dof_pos
        obs += (2 * torch.rand_like(obs) - 1) * noise_scale * global_noise_level

    l_low=env.default_dof_pos-env.dof_pos_limits[:,0]
    l_up=env.dof_pos_limits[:,1]-env.default_dof_pos
    norm_scale=torch.max(l_low,l_up)
    return obs / norm_scale
