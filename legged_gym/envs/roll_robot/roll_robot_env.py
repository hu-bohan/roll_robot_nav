from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import torch
import numpy as np
from isaacgym import gymapi, gymtorch

class RollRobotEnv(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.camera_handles = []#TODO 这俩东西在config里有定义吗？
        self.camera_tensors = [] 

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
        camera_props.horizontal_fov = self.cfg.camera.horizontal_fovss

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
        # TODO 暂时保持原样，先把相机加上看看报不报错
        return super().step(actions)