import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
class ForceTorquePlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle('Real Time End-Effector Data')
        self.setGeometry(100, 100, 600, 800)

        # 创建主布局
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)

        # 创建第一个绘图窗口（位置数据）
        self.force_graph = pg.PlotWidget()
        self.layout.addWidget(self.force_graph)
        self.force_graph.setTitle("Real Time external force at body", color="b", size="20pt")
        self.force_graph.setLabel('left', 'Force')
        self.force_graph.setLabel('bottom', 'Time')
        self.force_graph.setBackground('w')
        self.force_graph.addLegend()

        # 初始化位置数据
        self.x = list(range(100))  # x轴数据（时间）
        self.f1 = [0] * 100  # x_pos[0]
        self.f2 = [0] * 100  # x_pos[1]
        self.f3 = [0] * 100  # x_pos[2]

        # 绘制初始位置曲线
        self.curve1 = self.force_graph.plot(self.x, self.f1, pen=pg.mkPen('r', width=2), name='X force')
        self.curve2 = self.force_graph.plot(self.x, self.f2, pen=pg.mkPen('g', width=2), name='Y force')
        self.curve3 = self.force_graph.plot(self.x, self.f3, pen=pg.mkPen('b', width=2), name='Z force')

        self.torque_graph = pg.PlotWidget()
        self.layout.addWidget(self.torque_graph)
        self.torque_graph.setTitle("Real Time external torque at body", color="b", size="20pt")
        self.torque_graph.setLabel('left', 'torque')
        self.torque_graph.setLabel('bottom', 'Time')
        self.torque_graph.setBackground('w')
        self.torque_graph.addLegend()

        # 初始化加速度数据
        self.t1 = [0] * 100  # x_vel_linear[0]
        self.t2 = [0] * 100  # x_vel_linear[1]
        self.t3 = [0] * 100  # x_vel_linear[2]

        # 绘制初始速度曲线
        self.curve4 = self.torque_graph.plot(self.x, self.t1, pen=pg.mkPen('r', width=2), name='X torque')
        self.curve5 = self.torque_graph.plot(self.x, self.t2, pen=pg.mkPen('g', width=2), name='Y torque')
        self.curve6 = self.torque_graph.plot(self.x, self.t3, pen=pg.mkPen('b', width=2), name='Z torque')

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)
    def connect_thread(self, simulation_thread):
        # 设置定时器，每隔100ms更新一次数据
        # 启动仿真线程
        simulation_thread.data_updated.connect(self.update_data)  # 连接信号槽
        simulation_thread.start()

    def update_plot(self):
        self.curve1.setData(self.x, self.f1)
        self.curve2.setData(self.x, self.f2)
        self.curve3.setData(self.x, self.f3)

        self.curve4.setData(self.x, self.t1)
        self.curve5.setData(self.x, self.t2)
        self.curve6.setData(self.x, self.t3)

    def update_data(self, f_x, f_y, f_z, t_x, t_y, t_z):
        self.f1.append(f_x)
        self.f1.pop(0)
        self.f2.append(f_y)
        self.f2.pop(0)
        self.f3.append(f_z)
        self.f3.pop(0)

        self.t1.append(t_x)
        self.t1.pop(0)
        self.t2.append(t_y)
        self.t2.pop(0)
        self.t3.append(t_z)
        self.t3.pop(0)