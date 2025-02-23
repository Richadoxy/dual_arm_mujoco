

class TrajectoryMonitor:
    def __init__(self, pos_traj, ori_traj):
        """
        初始化轨迹监控器
        :param pos_traj: 位置轨迹（Nx3数组）
        :param ori_traj: 姿态轨迹（Nx4数组）
        """
        self.pos_traj = pos_traj
        self.ori_traj = ori_traj
        self.current_idx = 0  # 当前目标点索引

    def get_current_target(self):
        """获取当前目标点（位置+姿态）"""
        return self.pos_traj[self.current_idx], self.ori_traj[self.current_idx]

    def advance(self):
        """推进到下一个目标点"""
        if self.current_idx < len(self.pos_traj) - 1:
            self.current_idx += 1
            return True
        return False