import numpy as np

from planning import ArcTrajectory
from robot import Robot_mj, init_actuator
import mujoco
import os
import numpy
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 XML 文件的绝对路径
xml_path = os.path.join(current_dir, "../../model/scene.xml")

def SortingTrajGeneration():
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    init_actuator(model, data, "pos_mode")
    right_arm = Robot_mj(model, data, 7, "end_effector_right")
    left_arm = Robot_mj(model, data, 7, "end_effector_left")
    q_init = [right_arm.get_qinfo()[0], left_arm.get_qinfo()[0]]
    init_pos = [right_arm.fk()[0], left_arm.fk()[0]]
    init_ori = [right_arm.fk()[2], left_arm.fk()[2]]

    # 获取目标点
    object_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder")
    object_pos = data.geom_xpos[object_index]

    grab_pos = object_pos + np.array([0.0, 0.0, 0.1])  # 右臂抓取点
    grab_ori = np.array([0, 1, 0, 0])

    ready_pos = grab_pos + np.array([0.0, -0.7, 0])  # 左臂准备点
    ready_ori = np.array([0, 1, 0, 0])
    pick_pos = ready_pos + np.array([0.0, 0.3, 0])  # 左臂拾取点
    pick_ori = np.array([0.7071068, -0.7071068, 0, 0])

    deliver_pos = grab_pos + np.array([0.0, -0.3, 0])  # 右臂放置点
    deliver_ori = np.array([0.7071068, 0.7071068, 0, 0])

    goal_pos = np.array([0.45, -0.35, 0.2])
    goal_ori = np.array([0, 1, 0, 0])



    # 生成轨迹
    total_time = 2
    dt = 0.005
    right_arm_grab_traj = ArcTrajectory(
        robot=right_arm,
        start_point=init_pos[0],
        # mid_point=np.array([0.25569317, 0.65391336, 0.62723582]),
        mid_point=(init_pos[0] + grab_pos) / 2 + np.array([0.001, 0.001, 0.001]),

        end_point=grab_pos,
        start_quat=init_ori[0],
        # mid_quat= np.array([0.827, -0.4776688710702888, 0.2559260646188239, -0.1477596450711065]),
        end_quat=grab_ori,
        total_time=total_time,
        dt=dt
    )
    time_stemp_phase1, pos_traj_grab, ori_traj_grab = right_arm_grab_traj.generate_trajectory()
    left_arm_ready_traj = ArcTrajectory(
        robot=left_arm,
        start_point=init_pos[1],
        mid_point=(init_pos[1] + ready_pos) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=ready_pos,
        start_quat=init_ori[1],
        # mid_quat=np.array([-0.245, 0.784, -0.202, 0.532]),
        end_quat=ready_ori,
        total_time=total_time,
        dt=dt
    )
    _, pos_traj_ready, ori_traj_ready = left_arm_ready_traj.generate_trajectory()
    right_arm_deliver_traj = ArcTrajectory(
        robot=right_arm,
        start_point=grab_pos,
        mid_point=(grab_pos + deliver_pos) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=deliver_pos,
        start_quat=grab_ori,
        end_quat=deliver_ori,
        total_time=total_time,
        dt=dt
    )
    time_stemp_phase2, pos_traj_deliver, ori_traj_deliver = right_arm_deliver_traj.generate_trajectory()

    left_arm_pick_traj = ArcTrajectory(
        robot=left_arm,
        start_point=ready_pos,
        mid_point=(ready_pos + pick_pos) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=pick_pos,
        start_quat=ready_ori,
        end_quat=pick_ori,
        total_time=total_time,
        dt=dt
    )
    _, pos_traj_pick, ori_traj_pick = left_arm_pick_traj.generate_trajectory()

    left_arm_goal_traj = ArcTrajectory(
        robot=left_arm,
        start_point=pick_pos,
        mid_point=(pick_pos + goal_pos) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=goal_pos,
        start_quat=pick_ori,
        end_quat=goal_ori,
        total_time=total_time,
        dt=dt
    )

    _, pos_traj_goal, ori_traj_goal = left_arm_goal_traj.generate_trajectory()
    # left_arm_goal_traj.plot_orientation_along_trajectory()

    left_arm_home_traj = ArcTrajectory(
        robot=left_arm,
        start_point=goal_pos,
        mid_point=(goal_pos + init_pos[1]) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=init_pos[1],
        start_quat=goal_ori,
        end_quat=init_ori[1],
        total_time=total_time,
        dt=dt
    )

    _, pos_traj_home_left, ori_traj_home_left = left_arm_home_traj.generate_trajectory()

    right_arm_home_traj = ArcTrajectory(
        robot=left_arm,
        start_point=deliver_pos,
        mid_point=(deliver_pos + init_pos[0]) / 2 + np.array([0.001, 0.001, 0.001]),
        end_point=init_pos[0],
        start_quat=deliver_ori,
        end_quat=init_ori[0],
        total_time=total_time,
        dt=dt
    )

    _, pos_traj_home_right, ori_traj_home_right = right_arm_home_traj.generate_trajectory()

    # 合并轨迹
    full_right_pos = np.vstack([pos_traj_grab, pos_traj_deliver])
    full_right_ori = np.vstack([ori_traj_grab, ori_traj_deliver])
    full_left_pos = np.vstack([pos_traj_ready, pos_traj_pick, pos_traj_goal])
    full_left_ori = np.vstack([ori_traj_ready, ori_traj_pick, ori_traj_goal])
    # full_left_pos = np.vstack([pos_traj_ready, pos_traj_pick])
    # full_left_ori = np.vstack([ori_traj_ready, ori_traj_pick])
    return full_right_pos, full_right_ori, full_left_pos, full_left_ori
