import signal
import time
import sys
from PyQt5 import QtWidgets
import mujoco
import mujoco.viewer as viewers
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from robot import Robot_mj, init_actuator
from qt_gui import Traj3DPlot
from robot import Robot_mj, init_actuator
from planning import SortingTrajGeneration, TrajectoryMonitor
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 XML 文件的绝对路径
xml_path = os.path.join(current_dir, "../model/scene.xml")
# 加载 XML 文件

class SimulationThread(QThread):
    # 定义一个信号，用于传递数据（三个末端位置）
    data_updated = pyqtSignal(float, float, float, float, float, float, float, float, float, float, float, float, float, float)  # 14个数据，前七个为右臂的末端位姿，后期个为左臂

    def run(self):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        init_actuator(model, data, "pos_mode")
        right_arm = Robot_mj(model, data, 7, "end_effector_right")
        left_arm = Robot_mj(model, data, 7, "end_effector_left")
        q_init = [right_arm.get_qinfo()[0], left_arm.get_qinfo()[0]]
        init_pos = [right_arm.fk()[0], left_arm.fk()[0]]
        init_ori = [right_arm.fk()[2], left_arm.fk()[2]]

        full_right_pos, full_right_ori, full_left_pos, full_left_ori = SortingTrajGeneration()
        # 初始化轨迹监控器
        right_monitor = TrajectoryMonitor(full_right_pos, full_right_ori)
        left_monitor = TrajectoryMonitor(full_left_pos, full_left_ori)
        with viewers.launch_passive(model, data) as viewer:
            step = 0
            step_time = model.opt.timestep
            q_result = [q_init[0], q_init[1]]
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
                time.sleep(step_time)

                # 记录实际轨迹
                pos_right, _, quat_right = right_arm.fk()
                pos_left, _, quat_left = left_arm.fk()
                self.data_updated.emit(
                    pos_right[0], pos_right[1], pos_right[2],
                    quat_right[0], quat_right[1], quat_right[2], quat_right[3],
                    pos_left[0], pos_left[1], pos_left[2],
                    quat_left[0], quat_left[1], quat_left[2], quat_left[3]
                )

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

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    window = Traj3DPlot()
    sim_thread = SimulationThread()
    window.connect_thread(sim_thread)
    window.show()
    sys.exit(app.exec_())