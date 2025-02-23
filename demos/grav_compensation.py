import signal
import time
import sys
from PyQt5 import QtWidgets
import mujoco
import mujoco.viewer as viewer
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from robot import Robot_mj, init_actuator
from qt_gui import ForceTorquePlot
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 XML 文件的绝对路径
xml_path = os.path.join(current_dir, "../model/scene.xml")
# 加载 XML 文件

class SimulationThread(QThread):
    # 定义一个信号，用于传递数据（三个末端位置）
    data_updated = pyqtSignal(float, float, float, float, float, float)  # 发送三个浮点数

    def run(self):
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        # 初始化位置控制模式
        init_actuator(model, data, ctrl_mode="tqe_mode")
        right_arm = Robot_mj(model, data, 7, "end_effector_right")
        left_arm = Robot_mj(model, data, 7, "end_effector_left")
        num_arm = 2

        right_arm_damp = [0.3, 0.7, 0.2, 0.3, 0.01, 0.02, 0.01]
        #right_arm_damp = [0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00]
        left_arm_damp = [0.3, 0.7, 0.2, 0.3, 0.01, 0.02, 0.01]
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_time = model.opt.timestep
            step = 0
            damp = [right_arm_damp, left_arm_damp]
            arms = [right_arm, left_arm]
            cg = [0, 0]
            vel = [0, 0]
            while viewer.is_running():
                for i in range(2):
                    _,vel[i],_ = arms[i].get_qinfo()
                    cg[i] = arms[i].coriolis_gravity()
                    data.ctrl[arms[i].index] = cg[i] - damp[i] * vel[i]
                    step += 1
                    # 控制仿真速度
                time.sleep(step_time)
                mujoco.mj_step(model, data)
                viewer.sync()
                for body_id in range(model.nbody):
                    xfrc = data.xfrc_applied[body_id]
                    if np.any(xfrc != 0):  # 如果外力不为零
                        # 发送外力数据和 body ID
                        self.data_updated.emit(
                            xfrc[0], xfrc[1], xfrc[2],  # 力的分量
                            xfrc[3], xfrc[4], xfrc[5],  # 力矩的分量
                        )
                        break  # 只发送第一个检测到的外力不为零的 body


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app = QtWidgets.QApplication(sys.argv)
    window = ForceTorquePlot()
    sim_thread = SimulationThread()
    window.connect_thread(sim_thread)
    window.show()
    sys.exit(app.exec_())