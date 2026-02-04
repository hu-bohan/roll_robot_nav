from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class RollRobotCfg(LeggedRobotCfg):

    #对已有类进行修改
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.3] # x,y,z [m]

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'hipdriver1': 0.,
            'hipdriver2': 0.,
            'hipdriver3': 0.,
            'hipdriver4': 0.,
            'hipdriver5': 0.,
            'hipdriver6': 0.,

            'thighdriver1': 0.,  
            'thighdriver2': 0.,  
            'thighdriver3': 0.,
            'thighdriver4': 0.,
            'thighdriver5': 0.,
            'thighdriver6': 0.,

            'shankdriver1': -0.2,
            'shankdriver2': -0.2,
            'shankdriver3': -0.2, 
            'shankdriver4': -0.2, 
            'shankdriver5': -0.2, 
            'shankdriver6': -0.2, 
        }

        default_joint_pos= [value for key, value in default_joint_angles.items()]
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/roll_robot/urdf/roll_robot_r.urdf'
        name = "roll_robot"
        foot_name = "foot"
        hip_name = "hip"
        thigh_name = "thigh"
        shank_name = "shank"
        base_name = "base_link"
        penalize_contacts_on = ["shank"]
        penalize_self_collision = ["hip", "thigh"]
        terminate_after_contacts_on = ["base_link"]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'hipdriver': 30,
            'thighdriver': 30,
            'shankdriver': 30,
        }  # [N*m/rad]
        damping = {
            'hipdriver': 1.5,
            'thighdriver': 1.5,
            'shankdriver': 1.5,
        }     # [N*m*s/rad]
        torque_limits = {
            'hipdriver': 12.,
            'thighdriver': 12.,
            'shankdriver': 12.,
        }     # [N*m]
        action_scale = 1.0
        decimation = 4

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 1.0
            ang_vel = 0.1
            dof_pos = 1.0
            dof_vel = 0.1
            height_measurements = 1.0
        
        clip_observations = 10.
        clip_actions = 1.

    class env(LeggedRobotCfg.env):
        num_envs = 64
        num_actions = 18  #TODO 这里在修改输出为high level commands以后要改为3
        # TODO 更新观测维度：深度图像素数 + 目标向量(3) + 自身状态等
        num_observations = 253# TODO 这里只是举例,等正式确定观测维度以后记得修改

    
    #新添加的类
    class camera:
        # Realsense D435 参数
        width = 128 # 1.分辨率
        height = 72
        horizontal_fov = 85.2 # 2.视场角 
        position = [0.3, 0.0, 0.2]   # 3. 安装位置 (相对于 base_link) [x, y, z]
        rotation = [0.0, 0.0, 0.0] # 4. 安装角度 [roll, pitch, yaw] (弧度)


class RollRobotCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'roll_robot_nav'
        experiment_name = 'navigation_task'