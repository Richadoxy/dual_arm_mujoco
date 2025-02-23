import mujoco
import mujoco.viewer as viewer
import numpy as np
from robot import Robot_mj, init_actuator
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造 XML 文件的绝对路径
xml_path = os.path.join(current_dir, "../model/scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# 初始化位置控制模式
init_actuator(model, data, ctrl_mode="pos_mode")
right_arm = Robot_mj(model,data,7, "end_effector_right")
left_arm = Robot_mj(model,data, 7, "end_effector_left")

mujoco.mj_resetDataKeyframe(model, data, 4)
mujoco.mj_forward(model, data)

left_pos, _, left_quat = left_arm.fk()
print(' end_effector_left: ', left_pos, left_quat)
right_pos, _, right_quat = right_arm.fk()
print('end_effector_right: ', right_pos, right_quat)


# hull = right_arm.build_workspace_hull()
#
# # 检查点是否在工作空间内
# point_to_check = right_pos
# if right_arm.is_point_in_workspace_hull(point_to_check, hull):
#     print("点在机械臂的工作空间内")
# else:
#     print("点不在机械臂的工作空间内")
viewer.launch(model,data)