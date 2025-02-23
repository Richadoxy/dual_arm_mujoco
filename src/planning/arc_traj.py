import numpy as np
import matplotlib.pyplot as plt
from robot import Robot_mj

class ArcTrajectory:
    def __init__(self,robot,  start_point, mid_point, end_point, start_quat, end_quat, total_time, dt, mid_quat=None):
        """
        初始化圆形轨迹类
        :param start_point: 起始点，形状为 (3,) 的 numpy 数组
        :param mid_point: 中间点，形状为 (3,) 的 numpy 数组
        :param end_point: 终点，形状为 (3,) 的 numpy 数组
        :param start_quat: 起始四元数，形状为 (4,) 的 numpy 数组
        :param end_quat: 结束四元数，形状为 (4,) 的 numpy 数组
        :param total_time: 总时间（秒）
        :param dt: 时间步长（秒）
        :param mid_quat: 中间四元数，形状为 (4,) 的 numpy 数组，默认为 None
        """
        self.start_point = start_point
        self.mid_point = mid_point
        self.end_point = end_point
        self.robot = robot
        self.start_quat = np.array(start_quat)
        self.end_quat = np.array(end_quat)
        self.mid_quat = np.array(mid_quat) if mid_quat is not None else None

        self.total_time = total_time
        self.dt = dt
        self.time = np.arange(0, self.total_time, dt)
        self.center = None
        self.radius = None
        self.trajectory = None
        self.velocity = None
        self.acceleration = None
        self.quaternions = None

    def calculate_circle(self):
        """
        计算通过三个点的圆
        """
        p1, p2, p3 = self.start_point, self.mid_point, self.end_point

        # 计算向量
        v1 = p2 - p1
        v2 = p3 - p1

        v1n = v1/np.linalg.norm(v1)
        v2n = v2/np.linalg.norm(v2)

        nv = np.cross(v1n, v2n)

        # 检查点是否共线

        if np.allclose(nv, [0, 0, 0]):
            raise ValueError("三个点共线，无法生成圆形轨迹")

        # 计算圆心

        u = v1n
        w = np.cross(v2, v1) / np.linalg.norm(np.cross(v2, v1))
        v = np.cross(w, u)

        bx = np.dot(v1, u)
        cx = np.dot(v2, u)
        cy = np.dot(v2, v)
        h = ((cx - bx /2) ** 2 + cy ** 2 - (bx/2) ** 2)/(2 * cy)
        self.center = p1 + bx/2 * u + h*v

        # 计算半径
        self.radius = np.linalg.norm(self.center - p1)

    def quintic_polynomial(self, t, t0, tf, q0, qf, v0=0, vf=0, a0=0, af=0):
        """
        五次多项式插值
        :param t: 当前时间
        :param t0: 起始时间
        :param tf: 结束时间
        :param q0: 起始位置
        :param qf: 结束位置
        :param v0: 起始速度
        :param vf: 结束速度
        :param a0: 起始加速度
        :param af: 结束加速度
        :return: 位置、速度、加速度
        """
        T = tf - t0
        tau = (t - t0) / T
        tau2 = tau**2
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5

        # 位置
        q = q0 + (qf - q0) * (6 * tau5 - 15 * tau4 + 10 * tau3)
        # 速度
        v = (qf - q0) * (30 * tau4 - 60 * tau3 + 30 * tau2) / T
        # 加速度
        a = (qf - q0) * (120 * tau3 - 180 * tau2 + 60 * tau) / (T**2)

        return q, v, a

    def _cubic_quaternion_interpolation(self, t):
        """
        根据是否提供中间四元数，进行分段或单段三次多项式四元数插值
        """
        if self.mid_quat is None:
            # 单段插值：start_quat -> end_quat
            t_normalized = t / self.total_time
            q_start = self.start_quat.copy()
            q_end = self.end_quat.copy()

            # 处理四元数符号一致性
            if np.dot(q_start, q_end) < 0:
                q_end = -q_end

            # 计算插值参数
            theta = np.arccos(np.clip(np.dot(q_start, q_end), -1.0, 1.0))
            if theta < 1e-6:
                return q_start.copy()

            # 三次多项式参数化（s(t) = 3t^2 - 2t^3）
            s = 3 * t_normalized**2 - 2 * t_normalized**3

            # 球面线性插值
            interpolated = (np.sin((1 - s)*theta) * q_start + np.sin(s*theta) * q_end) / np.sin(theta)

        else:
            # 分段插值：start_quat -> mid_quat -> end_quat
            half_time = self.total_time / 2

            # 确定当前时间段和对应的起始、结束四元数
            if t <= half_time:
                t_segment = t
                duration = half_time
                q_start = self.start_quat.copy()
                q_end = self.mid_quat.copy()
            else:
                t_segment = t - half_time
                duration = self.total_time - half_time
                q_start = self.mid_quat.copy()
                q_end = self.end_quat.copy()

            # 处理四元数符号一致性
            if np.dot(q_start, q_end) < 0:
                q_end = -q_end

            # 计算插值参数
            theta = np.arccos(np.clip(np.dot(q_start, q_end), -1.0, 1.0))
            if theta < 1e-6:
                return q_start.copy()

            # 三次多项式参数化（s(t) = 3t^2 - 2t^3）
            t_normalized = t_segment / duration
            s = 3 * t_normalized**2 - 2 * t_normalized**3

            # 球面线性插值
            interpolated = (np.sin((1 - s)*theta) * q_start + np.sin(s*theta) * q_end) / np.sin(theta)

        # 确保实部非负
        if interpolated[0] < 0:
            interpolated = -interpolated

        return interpolated

    def repair_trajectory_point(self, pos, quat, q_init,max_attempts=10, step_size=0.01):
        """
        基于梯度下降的轨迹点修复
        :param pos: 目标位置 (3,)
        :param quat: 目标姿态 (4,)
        :param robot: 机器人实例
        :param q_init: 初始关节角度
        :param max_attempts: 最大尝试次数
        :param step_size: 梯度步长
        :return: 修复后的位置和姿态
        """
        best_pos = pos.copy()
        best_quat = quat.copy()
        for _ in range(max_attempts):
            # 计算当前误差
            q, iter = self.robot.ik_damping(best_pos, best_quat, q_init)
            if iter<49:
                return best_pos, best_quat
            best_pos+=np.array([0.001, -0.001, -0.001])

        return q



    def generate_trajectory(self):
        """
        生成圆形轨迹
        """
        # hull = self.robot.build_workspace_hull()
        if self.center is None or self.radius is None:
            self.calculate_circle()
        p1, p2, p3, pc = self.start_point, self.mid_point, self.end_point, self.center

        # 计算A, B, C
        A = (p2[1] - p1[1]) * (p3[2] - p2[2]) - (p2[2] - p1[2]) * (p3[1] - p2[1])
        B = (p2[2] - p1[2]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[2] - p2[2])
        C = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

        # 计算K
        K = np.sqrt(A ** 2 + B ** 2 + C ** 2)

        # 计算单位法向量a
        a = np.array([A, B, C]) / K

        # 计算圆心到p1的单位向量n
        n = (p1 - self.center) / self.radius

        # 计算交叉积
        o = np.cross(a, n)

        # 构造变换矩阵T
        T = np.vstack([np.column_stack([n, o, a, pc]), np.array([0, 0, 0, 1])])

        # 求转换后的点q1, q2, q3
        q1 = np.dot(np.linalg.inv(T), np.append(p1, 1))
        q2 = np.dot(np.linalg.inv(T), np.append(p2, 1))
        q3 = np.dot(np.linalg.inv(T), np.append(p3, 1))

        # 计算角度theta13和theta12
        theta13 = np.arctan2(q3[1], q3[0])
        if q3[1] < 0:
            theta13 += 2 * np.pi

        theta12 = np.arctan2(q2[1], q2[0])
        if q2[1] < 0:
            theta12 += 2 * np.pi

        # 生成轨迹点
        num_points = len(self.time)
        self.trajectory = np.zeros((num_points, 4))
        self.velocity = np.zeros((num_points, 3))
        self.acceleration = np.zeros((num_points, 3))
        self.quaternions = np.zeros((num_points, 4))
        for i, t in enumerate(self.time):
            # 使用五次多项式插值计算角度
            theta, omega, alpha = self.quintic_polynomial(t, 0, self.total_time, 0, theta13, 0, 0, 0, 0)

            # 计算位置
            self.trajectory[i] = np.dot(T, np.array([self.radius * np.cos(theta), self.radius * np.sin(theta), 0, 1]))


            # 计算速
            self.quaternions[i] = self._cubic_quaternion_interpolation(t)

        # for i in range(len(self.trajectory)):
        #     pos = self.trajectory[i, :3]
        #     print(pos)
        #     if self.robot.is_point_in_workspace_hull(pos,hull):
        #         print("点在机械臂的工作空间内")
        #     else:
        #         print("点不在机械臂的工作空间内")


        # self.trajectory = np.array([p for p, _ in repaired_traj])
        # self.quaternions = np.array([q for _, q in repaired_traj])
        return self.time, self.trajectory[:,:3], self.quaternions

    def save_trajectory(self, filename):
        """
        保存轨迹到 .npz 文件
        :param filename: 文件名（不需要扩展名）
        """
        if self.trajectory is None:
            raise ValueError("未生成轨迹，请先调用 generate_trajectory()")
        np.savez(filename, time=self.time, positions=self.trajectory[:,0:3], quaternions=self.quaternions)

    def plot_time_trajectory(self):
        """
        绘制轨迹的位置、速度、加速度随时间的变化
        """
        if self.trajectory is None:
            raise ValueError("未生成轨迹，请先调用 generate_trajectory()")

        plt.figure(figsize=(12, 10))

        # 绘制位置
        plt.subplot(3, 1, 1)
        plt.plot(self.time, self.trajectory[:, 0], label='X Position')
        plt.plot(self.time, self.trajectory[:, 1], label='Y Position')
        plt.plot(self.time, self.trajectory[:, 2], label='Z Position')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.title('Position vs Time')
        plt.legend()

        # 绘制姿态（四元数）
        plt.subplot(3, 1, 2)
        plt.plot(self.time, self.quaternions[:, 0], label='Quat W')
        plt.plot(self.time, self.quaternions[:, 1], label='Quat X')
        plt.plot(self.time, self.quaternions[:, 2], label='Quat Y')
        plt.plot(self.time, self.quaternions[:, 3], label='Quat Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Quaternion')
        plt.title('Quaternion vs Time')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()


    def plot_orientation_along_trajectory(self, step=50, axis_length=0.1):
        """
        在轨迹上绘制姿态的坐标系表示
        :param step: 每隔多少点绘制一个坐标系（步长）
        :param axis_length: 坐标轴长度（米）
        """
        if self.trajectory is None or self.quaternions is None:
            raise ValueError("未生成轨迹或姿态数据，请先调用 generate_trajectory()")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制主轨迹
        ax.plot(self.trajectory[:, 0],
                self.trajectory[:, 1],
                self.trajectory[:, 2],
                label='Trajectory',
                color='blue',
                alpha=0.5)
        ax.scatter([self.start_point[0]],
                   [self.start_point[1]],
                   [self.start_point[2]], color='red', label='start Points')

        ax.scatter([self.mid_point[0]],
                   [self.mid_point[1]],
                   [self.mid_point[2]], color='yellow', label='mid Points')
        ax.scatter([self.end_point[0]],
                   [self.end_point[1]],
                   [self.end_point[2]], color='black', label='end Points')
        # 四元数转旋转矩阵的辅助函数
        def quaternion_to_rotation_matrix(q):
            """将单位四元数转换为旋转矩阵"""
            w, x, y, z = q
            return np.array([
                [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
            ])

        # 遍历轨迹点（按步长间隔）
        for i in range(0, len(self.trajectory), step):
            pos = self.trajectory[i, 0:3]
            q = self.quaternions[i]

            # 归一化四元数
            q_normalized = q / np.linalg.norm(q)

            # 获取旋转矩阵
            R = quaternion_to_rotation_matrix(q_normalized)

            # 绘制坐标轴
            for col, color in zip(range(3), ['red', 'green', 'blue']):
                axis = R[:, col] * axis_length
                ax.quiver(pos[0], pos[1], pos[2],
                          axis[0], axis[1], axis[2],
                          color=color,
                          linewidth=1,
                          arrow_length_ratio=0.3,
                          length=axis_length)

        # 设置图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='X Axis'),
            Patch(facecolor='green', label='Y Axis'),
            Patch(facecolor='blue', label='Z Axis')
        ]
        ax.legend(handles=legend_elements)

        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory with Orientation Visualization')
        plt.show()

