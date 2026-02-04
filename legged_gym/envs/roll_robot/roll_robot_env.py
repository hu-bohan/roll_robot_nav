from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym.torch_utils import quat_rotate_inverse, normalize
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch
from . import observations

class RollRobotEnv(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.camera_handles = []
        self.camera_tensors = [] 
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.nav_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_observations, device=self.device)# 这里的 nav_obs_buf 专门存上层的观测

        # 3. 初始化目标点 
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_pos[:, 0] = 5.0 # 先测试：目标在前方 5米处


        # 1. 加载下层网络
        self.locomotion_agent = torch.jit.load('low_level_network/locomotion.pt').to(self.device) 
        self.locomotion_agent.eval() # 必须设为 eval 模式

        # 初始化用于下层输入的 Buffer
        # 假设 last_actions 是 18 维
        self.last_actions_for_low_level = torch.zeros(self.num_envs, 18, device=self.device, dtype=torch.float)


    def _create_envs(self):
        """
        覆盖父类的 _create_envs,添加相机 Sensor
        """
        # 1. 先调用父类方法，创建地形、机器人 Actor 等
        super()._create_envs()

        # 2. 定义相机属性 (D435)
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cfg.camera.width
        camera_props.height = self.cfg.camera.height
        camera_props.enable_tensors = True # 开启 GPU Tensor 加速
        camera_props.horizontal_fov = self.cfg.camera.horizontal_fov
        camera_props.horizontal_fov = self.cfg.camera.horizontal_fov

         # 定义安装位置的变换 (相对于绑定的 Body)
        local_transform = gymapi.Transform()
        local_transform.p = gymapi.Vec3(*self.cfg.camera.position)

         # 将欧拉角转换为四元数
        local_transform.r = gymapi.Quat.from_euler_zyx(
            self.cfg.camera.rotation[2], # Yaw
            self.cfg.camera.rotation[1], # Pitch
            self.cfg.camera.rotation[0]  # Roll
        )

         # 3. 遍历所有环境，为每个机器人添加相机
        for i in range(self.num_envs):
            env_handle = self.envs[i]
            actor_handle = self.actor_handles[i]
            
            # 创建相机 Sensor
            cam_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            self.camera_handles.append(cam_handle)
            
            # 找到机器人的 Base Link Handle
            body_handle = self.gym.find_actor_rigid_body_handle(
                env_handle, 
                actor_handle, 
                self.cfg.asset.base_name
            )
            
            # 将相机刚性绑定到 Base Link 上
            self.gym.attach_camera_to_body(
                cam_handle, 
                env_handle, 
                body_handle, 
                local_transform, 
                gymapi.FOLLOW_TRANSFORM
            )
            
            # 获取深度图的 GPU Tensor
            # 注意：我们在 Sim2Sim 避障中通常只用深度图 (IMAGE_DEPTH)
            tensor = self.gym.get_camera_image_gpu_tensor(
                self.sim, 
                env_handle, 
                cam_handle, 
                gymapi.IMAGE_DEPTH
            )

            # wrap_tensor 将 isaac gym 的 tensor 转换为 pytorch tensor
            torch_tensor = gymtorch.wrap_tensor(tensor)
            self.camera_tensors.append(torch_tensor)

    def step(self, actions):

        # --- 1. 将上层网络的 Action 转换为 Command ---
        # 映射 [-1, 1] 到实际速度范围
        clip_actions = self.cfg.normalization.clip_actions
        # 上层输出也可能需要 clip 一下防止跑飞
        actions = torch.clip(actions, -clip_actions, clip_actions)
        
        # 写入 env.commands (这就是桥梁！)
        # 下层网络读取 obs_commands 时会读这里的数据
        self.commands[:, 0] = actions[:, 0] * self.cfg.commands.ranges.lin_vel_x[1]
        self.commands[:, 1] = actions[:, 1] * self.cfg.commands.ranges.lin_vel_y[1]
        self.commands[:, 2] = actions[:, 2] * self.cfg.commands.ranges.ang_vel_yaw[1]

         # --- 2. 构造下层网络的输入 (Student Obs) ---
        
        obs_real_time = torch.cat((
            # 假设 use_dof_limit_normalize 根据你下层训练时的配置设定，这里假设用普通的 dof_pos
            observations.obs_dof_pos(self, add_noise=False), 
            observations.obs_dof_vel(self, add_noise=False),
            observations.obs_projected_gravity(self, add_noise=False),
            observations.obs_base_ang_vel(self, add_noise=False),
        ), dim=-1)

        # 构造完整的 student_obs
        # 注意：obs_last_actions(self) 默认读取 self.actions
        # 但在 LeggedRobot 的父类逻辑里，self.actions 通常指当前 step 的输入
        # 在这里，我们需要传入“上一次下层网络输出的电机指令”
        # 所以我们需要维护一个 self.last_actions_for_low_level
        
        student_obs = torch.concat((
            obs_real_time,
            self.last_actions_for_low_level, # 使用手动维护的上一帧动作
            observations.obs_commands(self), # 读取刚刚更新的 self.commands
        ), dim=-1)

        # --- 3. 下层网络推理 ---
        with torch.no_grad():
            # 输出的是电机指令 (actions)
            motor_actions = self.locomotion_agent(student_obs)
            
        # 更新 last_actions 用于下一帧
        self.last_actions_for_low_level = motor_actions.clone()

        # --- 4. 物理引擎步进 ---
        # 这里调用父类的 step (LeggedRobot.step)
        # 父类的 step 负责将 motor_actions 施加到关节上，并运行物理模拟
        # 并在内部更新 self.dof_pos, self.dof_vel 等 buffer
        
        # 【重要】为了兼容父类逻辑，我们将下层输出赋值给 self.actions
        # 父类内部会用到 self.actions 来计算 torque
        self.actions = motor_actions 
        
        # 执行物理模拟
        # 注意：这里会计算 reward，但这里的 reward 是下层的 reward
        # 我们需要在后面覆盖成上层的 reward
        super().step(self.actions)

        # --- 5. 计算上层网络的 Reward 和 Observation ---
        self.rew_buf[:] = 0. # 清空下层的 reward
        
        robot_pos = self.root_states[:, 0:3]
        target_dist = torch.norm(self.target_pos - robot_pos, p=2, dim=-1)
        is_reached = target_dist < 0.5
        # 如果到达目标，也触发 reset
        self.reset_buf |= is_reached 

        self.compute_nav_reward() 
        
        # 获取上层观测（相机 + 目标）
        self.obs_buf = self.compute_nav_observations() 

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_nav_observations(self): 
        """
        计算上层导航策略的观测值
        包含：压缩后的深度图(1D) + 目标向量
        """
        # 1. 刷新图形渲染 (必须调用，否则相机画面不更新)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        # 2. 处理深度图
        # stack 之后维度: (num_envs, height, width)
        depth_images = torch.stack(self.camera_tensors) 

        # Isaac Gym 的深度图：
        # 值通常是负数 (-distance)，单位是米。
        # -inf 表示无穷远。
        # 我们先取反变成正数，并处理 inf
        depth_images = -depth_images 

        # 将 inf 或极大值截断，超过 3米 的都算作 3米
        max_dist = 3.0
        depth_images[depth_images > max_dist] = max_dist
        depth_images[depth_images < 0.0] = max_dist # 处理可能的噪声

        # === 核心：压缩成 1D (Pseudo-LiDAR) ===
        # 维度变换: (num_envs, height, width) -> (num_envs, width)
        scan_1d, _ = torch.min(depth_images, dim=1)

        # 归一化：为了让网络更好训练，把 [0, 3] 映射到 [0, 1]
        scan_normalized = scan_1d / max_dist 

        self.gym.end_access_image_tensors(self.sim)

        # 3. 计算目标向量 (Target Vector)
        # 获取机器人当前的全局位置
        robot_pos = self.root_states[:, 0:3]
        robot_quat = self.root_states[:, 3:7]
        
        # 计算全局向量
        target_vec_global = self.target_pos - robot_pos
        
        # 【关键】将全局向量旋转到机器人的局部坐标系 (Base Frame)
        target_vec_local = quat_rotate_inverse(robot_quat, target_vec_global)

        # 结果维度: (num_envs, 128 + 3)
        self.nav_obs_buf = torch.cat((scan_normalized, target_vec_local), dim=-1)

        return self.nav_obs_buf

    def compute_nav_reward(self): 
        # 1. 获取位置信息
        robot_pos = self.root_states[:, 0:3]
        target_dist = torch.norm(self.target_pos - robot_pos, p=2, dim=-1)

        # 2. 奖励项：越靠近目标分越高 (Tracking Reward)
        # 使用指数形式通常比直接线性距离更稳定
        reward_tracking = torch.exp(-target_dist)

        # 3. 奖励项：到达目标 (Success Reward)
        # 设定一个阈值，比如 0.5米
        is_reached = target_dist < 0.5
        reward_success = is_reached.float() * 10.0 #注：由于达到目标点的同时也会触发reset，所以到达的奖励一定要比reset的惩罚高，才能有奖励作用

        # 4. 惩罚项：碰撞/摔倒 (Collision/Fall Penalty)
        # self.reset_buf 在 check_termination 中被更新
        # 注意：这里假设 reset_buf=1 代表摔倒或超时。
        # 如果你想区分摔倒和超时，可能需要去 check_termination 里细化
        reward_collision = self.reset_buf.float() * -5.0

        # === 汇总写入 rew_buf ===
        self.rew_buf = reward_tracking + reward_success + reward_collision


    def _reset_dofs(self, env_ids):
        # ... 原有的重置逻辑 ...
        super()._reset_dofs(env_ids)
        
        # 在重置时，随机生成新目标
        # 比如在机器人当前位置周围 3-5米 范围内生成
        self._resample_targets(env_ids)

    def _resample_commands(self, env_ids):
        """
        重写父类的方法。
        导航任务不需要随机采样“目标速度”，因为速度是由上层策略输出的。
        重置时我们将 commands 归零即可。
        """
        self.commands[env_ids] = 0.

    def _resample_targets(self, env_ids): 
        # 随机半径 [3, 5]
        r = torch.empty(len(env_ids), device=self.device).uniform_(3.0, 5.0)
        # 随机角度 [0, 2pi]
        theta = torch.empty(len(env_ids), device=self.device).uniform_(0, 2 * np.pi)
        
        # 计算偏移
        target_x = r * torch.cos(theta)
        target_y = r * torch.sin(theta)
        
        # 更新 target_pos (基于当前机器人位置或原点，看你地形设置)
        # TODO 稍作修改使之适配避障的地形
        # 获取每个环境的坐标原点 (env_origin)
        env_origins = self.env_origins[env_ids] # shape (len(env_ids), 3)

        # 加上随机偏移
        self.target_pos[env_ids, 0] = env_origins[:, 0] + target_x
        self.target_pos[env_ids, 1] = env_origins[:, 1] + target_y
        self.target_pos[env_ids, 2] = env_origins[:, 2] + 0.3