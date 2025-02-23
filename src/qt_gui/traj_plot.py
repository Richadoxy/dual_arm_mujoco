import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets, QtCore
import numpy as np
from planning.arc_traj import ArcTrajectory

class Traj3DPlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化窗口
        self.setWindowTitle('Dual-Arm Trajectory Visualization')
        self.setGeometry(100, 100, 800, 600)

        # 创建主容器和布局
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # 初始化轨迹数据存储
        self.right_arm_points = []
        self.left_arm_points = []

        # 创建单个3D视图


        # 颜色配置
        self.right_color = (0, 0, 1, 1)  # 蓝色表示右臂
        self.left_color = (1, 0, 0, 1)  # 红色表示左臂
        self.init_3d_view()
        # 轨迹显示长度限制（1000个点）
        self.max_points = 100

        # 定时刷新（可选）
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # 20Hz刷新

    def init_3d_view(self):
        """初始化单个3D视图"""
        # 创建3D视图
        self.view = gl.GLViewWidget(parent=self)
        self.view.setWindowTitle('Dual-Arm Trajectory')
        self.view.setCameraPosition(distance=2)  # 调整相机距离
        # self.view.setBackgroundColor('w')
        # 创建右臂轨迹
        self.right_plot = gl.GLScatterPlotItem()
        self.right_plot.setData(color=self.right_color, size=0.02, pxMode=False)
        self.view.addItem(self.right_plot)

        # 创建左臂轨迹
        self.left_plot = gl.GLScatterPlotItem()
        self.left_plot.setData(color=self.left_color, size=0.02, pxMode=False)
        self.view.addItem(self.left_plot)

        # 将视图添加到布局
        self.layout.addWidget(self.view)

        # 设置背景颜色为白色


        # 设置坐标轴
        axis = gl.GLAxisItem()
        axis.setSize(100, 100, 100)
        self.view.addItem(axis)

    def connect_thread(self, simulation_thread):
        """连接仿真线程的信号"""
        simulation_thread.data_updated.connect(self.update_data)
        simulation_thread.start()

    def update_data(self, right_px, right_py, right_pz,
                    right_qw, right_qx, right_qy, right_qz,
                    left_px, left_py, left_pz,
                    left_qw, left_qx, left_qy, left_qz):
        """接收并存储新数据点"""
        # 右臂新点
        new_right = np.array([right_px, right_py, right_pz])
        self.right_arm_points.append(new_right)

        # 左臂新点
        new_left = np.array([left_px, left_py, left_pz])
        self.left_arm_points.append(new_left)

        # 保持最大点数
        if len(self.right_arm_points) > self.max_points:
            self.right_arm_points.pop(0)
        if len(self.left_arm_points) > self.max_points:
            self.left_arm_points.pop(0)

    def update_plots(self):
        """更新3D视图"""
        # 右臂更新
        if self.right_arm_points:
            right_data = np.array(self.right_arm_points)
            self.right_plot.setData(pos=right_data)

        # 左臂更新
        if self.left_arm_points:
            left_data = np.array(self.left_arm_points)
            self.left_plot.setData(pos=left_data)

        # 强制刷新视图
        self.view.update()