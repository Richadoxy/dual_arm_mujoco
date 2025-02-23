import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer
import numpy as np
from robot import Robot_mj, init_actuator
from planning import TrajectoryMonitor
from planning.sorting_traj import SortingTrajGeneration
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 XML 文件的绝对路径
xml_path = os.path.join(current_dir, "../model/scene.xml")

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    init_actuator(model, data, "pos_mode")
    right_arm = Robot_mj(model, data, 7, "end_effector_right")
    left_arm = Robot_mj(model, data, 7, "end_effector_left")
    q_init = [right_arm.get_qinfo()[0], left_arm.get_qinfo()[0]]
    init_pos = [right_arm.fk()[0], left_arm.fk()[0]]
    init_ori = [right_arm.fk()[2], left_arm.fk()[2]]

    full_right_pos, full_right_ori, full_left_pos,full_left_ori = SortingTrajGeneration()
    # 初始化轨迹监控器
    right_monitor = TrajectoryMonitor(full_right_pos, full_right_ori)
    left_monitor = TrajectoryMonitor(full_left_pos, full_left_ori)

    with viewer.launch_passive(model, data) as viewer:
            step = 0
            step_time = model.opt.timestep
            arms = [right_arm, left_arm]
            q_result = [q_init[0], q_init[1]]
            actual_positions = [[], []]
            actual_quaternions = [[], []]
            iter = [0, 0]
            while viewer.is_running():
                # 获取当前目标点
                right_target = right_monitor.get_current_target()
                left_target = left_monitor.get_current_target()

                # 双臂独立求解
                q_result[0], _ = right_arm.ik_damping(
                    right_target[0], right_target[1], q_result[0]
                )
                q_result[1], _ = left_arm.ik_damping(
                    left_target[0], left_target[1], q_result[1]
                )

                # 更新控制指令
                data.ctrl[:14] = np.concatenate([q_result[0], q_result[1]])

                # 仿真步进
                mujoco.mj_step(model, data)
                viewer.sync()
                #time.sleep(step_time)

                # 记录实际轨迹
                pos_right, _, quat_right = right_arm.fk()
                pos_left, _, quat_left = left_arm.fk()
                actual_positions[0].append(pos_right)
                actual_quaternions[0].append(quat_right)
                actual_positions[1].append(pos_left)
                actual_quaternions[1].append(quat_left)

                # 推进条件检查
                if np.linalg.norm(pos_right - right_target[0]) < 0.5:  # 2cm阈值
                    right_monitor.advance()
                # else:
                #     print(f'right distance {np.linalg.norm(pos_right - right_target[0])}')
                if np.linalg.norm(pos_left - left_target[0]) < 0.5:
                    left_monitor.advance()
                # else:
                #     print(f'left distance {np.linalg.norm(pos_left - left_target[0])}')

                # 打印调试信息
                print(f"Step {step} | Right Target: {right_monitor.current_idx}/{len(full_right_pos)} | "
                      f"Left Target: {left_monitor.current_idx}/{len(full_left_pos)}")
                step += 1

                # 绘制轨迹
            actual_pos_right = np.array(actual_positions[0])
            actual_ori_right = np.array(actual_quaternions[0])
            actual_pos_left = np.array(actual_positions[1])
            actual_ori_left = np.array(actual_quaternions[1])

            fig, axs = plt.subplots(4, 2, figsize=(18, 20))
            for i in range(3):
                axs[i, 0].plot(full_right_pos[:, i], 'r--', label='Reference')
                axs[i, 0].plot(actual_pos_right[:, i], 'b-', label='Actual')
                axs[i, 0].set_title(f'Right Arm Position {["X", "Y", "Z"][i]}')
                axs[i, 0].legend()
                axs[i, 0].grid(True)

            for i in range(4):
                axs[i, 1].plot(full_right_ori[:, i], 'r--', label='Reference')
                axs[i, 1].plot(actual_ori_right[:, i], 'b-', label='Actual')
                axs[i, 1].set_title(f'Right Arm Orientation {["W", "X", "Y", "Z"][i]}')
                axs[i, 1].legend()
                axs[i, 1].grid(True)


            plt.tight_layout()
            plt.show()

            fig, axs = plt.subplots(4, 2, figsize=(18, 20))
            for i in range(3):
                axs[i, 0].plot(full_left_pos[:, i], 'r--', label='Reference')
                axs[i, 0].plot(actual_pos_left[:, i], 'b-', label='Actual')
                axs[i, 0].set_title(f'Left Arm Position {["X", "Y", "Z"][i]}')
                axs[i, 0].legend()
                axs[i, 0].grid(True)

            for i in range(4):
                axs[i, 1].plot(full_left_ori[:, i], 'r--', label='Reference')
                axs[i, 1].plot(actual_ori_left[:, i], 'b-', label='Actual')
                axs[i, 1].set_title(f'Left Arm Orientation {["W", "X", "Y", "Z"][i]}')
                axs[i, 1].legend()
                axs[i, 1].grid(True)

            plt.tight_layout()
            plt.show()