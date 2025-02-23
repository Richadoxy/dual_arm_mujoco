
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
class Robot_mj:
    #ee:end_effector
    def __init__(self, model, data,num_dof, ee):
        self.model = model
        self.data = data
        self.ee_name = ee
        if self.ee_name == "end_effector_right":
            self.index = np.arange(0, num_dof)
        elif self.ee_name == "end_effector_left":
            self.index = np.arange(num_dof, num_dof*2)
        else:
            self.index = 0
        mujoco.mj_forward(self.model, self.data)

    def get_qinfo(self):
        return self.data.qpos[self.index].copy(), self.data.qvel[self.index].copy(), self.data.qacc[self.index].copy()

    def get_joints_limits(self):
        joint_range = self.model.jnt_range[self.index]
        lower_limits = joint_range[:,0]
        upper_limits = joint_range[:,1]
        return lower_limits, upper_limits

    def fk(self):
        mujoco.mj_forward(self.model, self.data)
        #mujoco.mj_kinematics(self.model,self.data)
        site_id = self.model.site(self.ee_name).id
        pos = self.data.site_xpos[site_id].copy()
        ori = self.data.site_xmat[site_id].reshape([3, 3]).copy()
        rotation = R.from_matrix(ori)
        ori_quat_xyzw = rotation.as_quat()
        # 将四元数顺序从 [x, y, z, w] 转换为 [w, x, y, z]
        ori_quat_wxyz = [ori_quat_xyzw[3], ori_quat_xyzw[0], ori_quat_xyzw[1], ori_quat_xyzw[2]]
        return pos, ori, ori_quat_wxyz

    def jacobian(self):
        site_id = self.model.site(self.ee_name).id
        J_pos = np.zeros((3, self.model.nv))
        J_ori = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J_pos, J_ori, site_id)
        J_pos = J_pos[:, self.index]
        J_ori = J_ori[:, self.index]
        J_full = np.vstack([J_pos, J_ori])
        return J_full

    def mass_matrix(self):
        mass_matrix = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
        mass_matrix = mass_matrix[self.index, :][:, self.index]
        return mass_matrix

    def coriolis_gravity(self):
        return self.data.qfrc_bias[self.index]

    def cartesian_vel(self):
        # 获取末端执行器的 site_id
        site_id = self.model.site(self.ee_name).id

        # 初始化 Jacobian 矩阵
        J_pos = np.zeros((3, self.model.nv))  # 平移部分的 Jacobian
        J_ori = np.zeros((3, self.model.nv))  # 旋转部分的 Jacobian

        # 计算 Jacobian 矩阵
        mujoco.mj_jacSite(self.model, self.data, J_pos, J_ori, site_id)

        # 提取与关节速度相关的部分
        J_pos = J_pos[:, self.index]
        J_ori = J_ori[:, self.index]

        # 获取关节速度
        joint_velocity = self.data.qvel[self.index]

        # 计算笛卡尔空间中的线速度和角速度
        linear_velocity = J_pos @ joint_velocity  # 线速度
        angular_velocity = J_ori @ joint_velocity  # 角速度

        return linear_velocity, angular_velocity

    def jacobian_derivative(self, dt=1e-5):
        # 获取当前时间步的雅可比矩阵
        J_current = self.jacobian()

        # 保存当前状态
        qpos_prev = self.data.qpos.copy()
        qvel_prev = self.data.qvel.copy()

        # 前进一步模拟
        self.data.qpos[:] = qpos_prev + self.data.qvel * dt
        mujoco.mj_forward(self.model, self.data)

        # 获取下一步的雅可比矩阵
        J_next = self.jacobian()

        # 恢复之前的状态
        self.data.qpos[:] = qpos_prev
        self.data.qvel[:] = qvel_prev
        mujoco.mj_forward(self.model, self.data)

        # 计算雅可比矩阵的导数
        J_dot = (J_next - J_current) / dt

        return J_dot

    def cartesian_acc(self):
        # 获取雅可比矩阵
        J = self.jacobian()

        # 获取雅可比矩阵的导数
        J_dot = self.jacobian_derivative()

        # 获取关节速度和加速度
        joint_velocity = self.data.qvel[self.index]
        joint_acceleration = self.data.qacc[self.index]

        # 计算笛卡尔空间的加速度
        cartesian_acc = J @ joint_acceleration + J_dot @ joint_velocity

        return cartesian_acc

    def quaternion_error(self, q_des, q_current):
        """
        计算四元数姿态误差（转换为角速度）。

        参数：
        - q_des: 目标姿态 (四元数, 4维向量)
        - q_current: 当前姿态 (四元数, 4维向量)

        返回：
        - delta_omega: 角速度误差 (3维向量)
        """
        # 计算四元数误差：q_error = q_des * q_current^{-1}
        q_des_xyzw = [q_des[1], q_des[2], q_des[3], q_des[0]]
        q_current_xyzw = [q_current[1], q_current[2], q_current[3], q_current[0]]

        # 计算四元数误差：q_error = q_des * q_current^{-1}
        q_error = R.from_quat(q_des_xyzw) * R.from_quat(q_current_xyzw).inv()

        # 将误差转换为角速度
        delta_omega = q_error.as_rotvec()  # 转换为角速度
        return delta_omega

    def calculate_gradient(self, q):
        """优化后的限位规避梯度"""
        low_lim, upper_lim = self.get_joints_limits()
        margin = 0.1  # 缩小缓冲区间
        q_min = low_lim + margin
        q_max = upper_lim - margin
        q_mid = np.zeros_like(q)
        grad_com = (q_mid - q)
        grad_limit = np.zeros_like(q)
        grad = np.zeros_like(q)
        for i in range(len(q)):
            # 限位规避项（指数形式更平滑）
            if q[i] < q_min[i]:
                dist = q_min[i] - q[i]
                grad_limit[i] = 1.0 / (1 + np.exp(-10 * (dist - 0.05)))  # Sigmoid激活
            elif q[i] > q_max[i]:
                dist = q[i] - q_max[i]
                grad_limit[i] = -1.0 / (1 + np.exp(-10 * (dist - 0.05)))
        grad = grad_com +grad_limit
        return grad / (np.linalg.norm(grad) + 1e-8)

    def ik_damping(self, pos_des, quat_des, q_init, if_null_opt = False, max_iter=50, tol=1e-5):
        q_current = q_init.copy()
        low_lim, upper_lim = self.get_joints_limits()
        damping = 0.5  # 增大阻尼

        qpos_backup = self.data.qpos[self.index].copy()
        for iter in range(max_iter):
            # FK计算当前状态
            self.data.qpos[self.index] = q_current
            pos_current, _, quat_current = self.fk()

            # 计算任务空间误差
            delta_pos = pos_des - pos_current
            delta_quat = self.quaternion_error(quat_des, quat_current)
            delta_x = np.concatenate([delta_pos, delta_quat])

            if np.linalg.norm(delta_x) < tol:
                break

            # 雅可比矩阵与阻尼伪逆
            J = self.jacobian()
            lambda_sq = damping ** 2 * np.eye(6)
            J_damped = J.T @ np.linalg.pinv(J @ J.T + lambda_sq)
            delta_q_main = J_damped @ delta_x
            if if_null_opt == True:
                # 零空间优化项
                grad_h = self.calculate_gradient(q_current)
                P_null = np.eye(7) - J_damped @ J
                delta_null = P_null @ grad_h

                margin_ratio = np.clip(
                    (np.abs(q_current - low_lim) +
                     np.abs(upper_lim - q_current)) / 0.2,
                    0, 1
                )
                null_weight = 0.5 * np.mean(margin_ratio)
                q_current += delta_q_main + null_weight * delta_null
            #
            else:
                q_current += delta_q_main
            q_current = np.clip(q_current, low_lim, upper_lim)
            #print(f"Iter {iter}: Main={np.linalg.norm(delta_q_main):.6f}, Null={np.linalg.norm(delta_null):.6f}, Angle={np.degrees(np.arccos(cos_sim)):.1f}°")
        self.data.qpos[self.index] = qpos_backup
        mujoco.mj_forward(self.model, self.data)
        return q_current, iter

    def plot_workspace(self, num_samples=1000):
        """
        绘制机械臂的工作空间。

        参数：
        - num_samples: 采样的点数，默认为1000。
        """
        low_lim, upper_lim = self.get_joints_limits()
        qpos_backup = self.data.qpos[self.index].copy()

        # 随机采样关节角度
        sampled_positions = []
        for _ in range(num_samples):
            q_random = np.random.uniform(low_lim, upper_lim)
            self.data.qpos[self.index] = q_random
            mujoco.mj_forward(self.model, self.data)
            pos, _, _ = self.fk()
            sampled_positions.append(pos)

        # 恢复原始状态
        self.data.qpos[self.index] = qpos_backup
        mujoco.mj_forward(self.model, self.data)

        # 将采样点转换为数组
        sampled_positions = np.array(sampled_positions)

        # 绘制工作空间
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(sampled_positions[:, 0], sampled_positions[:, 1], sampled_positions[:, 2], c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Robot Workspace')
        plt.show()

