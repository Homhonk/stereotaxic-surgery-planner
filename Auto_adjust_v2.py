import numpy as np
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QWidget, QHBoxLayout
from mpl_toolkits.mplot3d import Axes3D
import stereo_camera_caculate_3d_v2
from step_shower import SurgeryStepsWidget
from ultralytics import YOLO
import threading
import time
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

import matplotlib
import serial
import sys
import os

matplotlib.rcParams['axes.edgecolor'] = '#333F4B'
matplotlib.rcParams['axes.linewidth'] = 1.2
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


class LevelingSystem:
    """自动调平系统"""

    def __init__(self):
        self.reference_points = None  # 参考位置（调平好的位置）
        self.current_points = None  # 当前位置
        self.transformation_matrix = None  # 变换矩阵
        self.translation_vector = None  # 平移向量
        self.rotation_angles = None  # 旋转角度 (roll, pitch, yaw)
        self.distance_deviation = None  # 距离偏差
        self.reference_file = 'reference_points.npy'
        self.load_reference_position()

    def save_reference_position(self):
        if self.reference_points is not None:
            np.save(self.reference_file, self.reference_points)

    def load_reference_position(self):
        if os.path.exists(self.reference_file):
            try:
                self.reference_points = np.load(self.reference_file)
                print("已从文件加载参考位置")
            except Exception as e:
                print(f"加载参考位置文件出错: {e}")
                self.reference_points = None

    def delete_reference_position(self):
        if os.path.exists(self.reference_file):
            try:
                os.remove(self.reference_file)
                print("已删除参考位置文件")
            except Exception as e:
                print(f"删除参考位置文件出错: {e}")

    def set_reference_position(self, points_3d):
        """设置参考位置（调平好的位置）"""
        if points_3d is None or len(points_3d) < 2:
            return False

        self.reference_points = points_3d.copy()
        self.save_reference_position()
        print("参考位置已设置并保存")
        return True

    def calculate_transformation(self, current_points, reference_points):
        """计算从当前位置到参考位置的变换关系"""
        if current_points is None or reference_points is None:
            return None, None, None, None

        if len(current_points) < 2 or len(reference_points) < 2:
            return None, None, None, None

        # 提取3D坐标
        current_coords = current_points[:, 0, :]
        reference_coords = reference_points[:, 0, :]

        # 计算质心
        current_centroid = np.mean(current_coords, axis=0)
        reference_centroid = np.mean(reference_coords, axis=0)

        # 中心化
        current_centered = current_coords - current_centroid
        reference_centered = reference_coords - reference_centroid

        # 使用SVD计算旋转矩阵
        try:
            H = current_centered.T @ reference_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # 确保旋转矩阵的行列式为正
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # 计算平移向量
            t = reference_centroid - R @ current_centroid

            # 构造变换矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t

            # 计算旋转角度 (roll, pitch, yaw)
            rotation = Rotation.from_matrix(R)
            angles = rotation.as_euler('xyz', degrees=True)

            # 计算距离偏差
            distances = []
            for i in range(min(len(current_coords), len(reference_coords))):
                current_transformed = R @ current_coords[i] + t
                distance = np.linalg.norm(current_transformed - reference_coords[i])
                distances.append(distance)

            avg_distance = np.mean(distances) if distances else 0

            return T, t, angles, avg_distance

        except Exception as e:
            print(f"变换计算错误: {e}")
            return None, None, None, None

    def update_current_position(self, points_3d):
        """更新当前位置并计算偏差"""
        if points_3d is None:
            return

        self.current_points = points_3d.copy()

        if self.reference_points is not None:
            self.transformation_matrix, self.translation_vector, \
                self.rotation_angles, self.distance_deviation = \
                self.calculate_transformation(self.current_points, self.reference_points)

    def get_leveling_info(self):
        """获取调平信息"""
        if self.reference_points is None:
            return "未设置参考位置"

        if self.transformation_matrix is None:
            return "无法计算变换关系"

        info = f"调平偏差信息:\n"
        if self.rotation_angles is not None:
            info += f"旋转角度: Roll={self.rotation_angles[0]:.2f}°, "
            info += f"Pitch={self.rotation_angles[1]:.2f}°, "
            info += f"Yaw={self.rotation_angles[2]:.2f}°\n"

        if self.translation_vector is not None:
            info += f"平移量: X={self.translation_vector[0]:.2f}mm, "
            info += f"Y={self.translation_vector[1]:.2f}mm, "
            info += f"Z={self.translation_vector[2]:.2f}mm\n"

        if self.distance_deviation is not None:
            info += f"平均距离偏差: {self.distance_deviation:.2f}mm"

        return info


class CoordinateDisplayUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("小鼠关键点3D坐标显示 - 自动调平系统")
        self.resize(1200, 800)
        # 初始化PID
        self.pid_pitch = PID(0.1, 0.01, 0.05)
        self.pid_roll = PID(0.1, 0.01, 0.05)
        # 主部件和布局
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)

        # 标题
        title_label = QtWidgets.QLabel("小鼠关键点3D坐标 - 自动调平系统")
        title_label.setStyleSheet("font-size:18px;font-weight:bold;")
        main_layout.addWidget(title_label)

        # 控制按钮 - 更紧凑的布局
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(10)  # 减少按钮间距
        
        self.set_reference_btn = QtWidgets.QPushButton("设置参考位置")
        self.set_reference_btn.setFixedWidth(120)
        self.set_reference_btn.clicked.connect(self.set_reference_position)
        btn_layout.addWidget(self.set_reference_btn)

        self.reset_reference_btn = QtWidgets.QPushButton("重置参考位置")
        self.reset_reference_btn.setFixedWidth(120)
        self.reset_reference_btn.clicked.connect(self.reset_reference_position)
        btn_layout.addWidget(self.reset_reference_btn)

        self.start_leveling_btn = QtWidgets.QPushButton("开始调平")
        self.start_leveling_btn.setFixedWidth(100)
        btn_layout.addWidget(self.start_leveling_btn)

        self.status_label = QtWidgets.QLabel("未设置参考位置")
        self.status_label.setStyleSheet("color:red;font-weight:bold;")
        btn_layout.addWidget(self.status_label)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # 左侧：表格、调平信息和注射路径信息
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        self.splitter.addWidget(left_widget)

        # 表格
        self.table = QtWidgets.QTableWidget(7, 4)
        self.table.setHorizontalHeaderLabels(["关键点", "X坐标(mm)", "Y坐标(mm)", "Z坐标(mm)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.keypoint_names = [
            'Bregma (前囟点)',
            'Lambda (后囟点)',
            'Point 1',
            'Point 2',
            'Point 3',
            'Point 4',
            'Point 5'
        ]
        for i, name in enumerate(self.keypoint_names):
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            for j in range(1, 4):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem("---"))
        left_layout.addWidget(self.table)
        
        # 加载注射路径信息

        # 调平信息
        leveling_group = QtWidgets.QGroupBox("调平信息")
        leveling_layout = QtWidgets.QVBoxLayout(leveling_group)
        self.leveling_text = QtWidgets.QTextEdit()
        self.leveling_text.setReadOnly(True)
        self.leveling_text.setFont(QtGui.QFont("Consolas", 10))
        leveling_layout.addWidget(self.leveling_text)
        left_layout.addWidget(leveling_group)

        # 右侧：3D可视化
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        self.splitter.addWidget(right_widget)

        # Matplotlib 3D 图
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        # 摄像头画面显示区域
        self.cam2_label = QtWidgets.QLabel("Camera 2")
        self.cam3_label = QtWidgets.QLabel("Camera 3")
        self.cam2_label.setFixedSize(640, 480)
        self.cam3_label.setFixedSize(640, 480)
        self.cam2_label.setStyleSheet("background-color: #cccccc;")
        self.cam3_label.setStyleSheet("background-color: #cccccc;")
        # 新建一个水平布局来横向放两个画面
        cam_layout = QtWidgets.QHBoxLayout()
        cam_layout.addWidget(self.cam2_label)
        cam_layout.addWidget(self.cam3_label)
        right_layout.addLayout(cam_layout)

        self.init_3d_plot()

        # 数据存储
        self.current_points = None
        self.reconstruction_system = None
        self.leveling_system = LevelingSystem()

        self.serial_port = None
        try:
            self.serial_port = serial.Serial('COM8', 9600, timeout=1)  # 修改为你的串口号
        except Exception as e:
            print(f"串口初始化失败: {e}")

        # 调平相关变量
        self.is_leveling = False
        self.leveling_thread = None

        self.start_leveling_btn.clicked.connect(self.start_leveling)

        # 启动定时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)

        # ----------- 新增：如果已加载参考点，自动更新状态文本 -----------
        if self.leveling_system.reference_points is not None:
            self.status_label.setText("参考位置已设置")
            self.status_label.setStyleSheet("color:green;font-weight:bold;")

    def set_reference_position(self):
        if self.current_points is not None:
            success = self.leveling_system.set_reference_position(self.current_points)
            if success:
                self.status_label.setText("参考位置已设置")
                self.status_label.setStyleSheet("color:green;font-weight:bold;")
                QtWidgets.QMessageBox.information(self, "成功", "参考位置设置成功！")
            else:
                QtWidgets.QMessageBox.critical(self, "错误", "设置参考位置失败，请确保检测到足够的关键点")
        else:
            QtWidgets.QMessageBox.critical(self, "错误", "当前没有检测到关键点，无法设置参考位置")

    def reset_reference_position(self):
        self.leveling_system.reference_points = None
        self.leveling_system.delete_reference_position()
        self.status_label.setText("未设置参考位置")
        self.status_label.setStyleSheet("color:red;font-weight:bold;")
        QtWidgets.QMessageBox.information(self, "重置", "参考位置已重置")

    def create_coordinate_display(self, parent):
        pass  # 已在 __init__ 中实现

    def create_leveling_info_display(self, parent):
        pass  # 已在 __init__ 中实现

    def create_3d_plot(self, parent):
        pass  # 已在 __init__ 中实现

    def init_3d_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('#F5F7FA')
        self.ax.grid(True, alpha=0.2)
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('小鼠关键点3D分布 - 调平系统')
        self.ax.set_xlim([-4, 4])
        self.ax.set_ylim([-4, 4])
        self.ax.set_zlim([-4, 4])
        # 删除中轴线设定
        self.ax.legend()
        self.canvas.draw()

    def update_coordinates(self, points_3d):
        if points_3d is None or len(points_3d) == 0:
            return
        if points_3d.ndim == 2 and points_3d.shape[1] == 3:
            points_3d = points_3d[:, np.newaxis, :]
        elif points_3d.ndim == 3 and points_3d.shape[1] != 1:
            points_3d = points_3d[:, :1, :]
        self.leveling_system.update_current_position(points_3d)
        for i in range(self.table.rowCount()):
            if i < len(points_3d):
                point = points_3d[i, 0, :]
                x, y, z = point[0], point[1], point[2]
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{x:.2f}"))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(f"{y:.2f}"))
                self.table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{z:.2f}"))
            else:
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem("---"))
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem("---"))
                self.table.setItem(i, 3, QtWidgets.QTableWidgetItem("---"))
        self.update_leveling_info()
        self.update_3d_plot(points_3d)

    def update_leveling_info(self):
        self.leveling_text.clear()
        if self.leveling_system.reference_points is None:
            self.leveling_text.append("未设置参考位置\n\n请先点击'设置参考位置'按钮")
            return
        if self.leveling_system.transformation_matrix is None:
            self.leveling_text.append("无法计算变换关系")
            return
        self.leveling_text.append("调平偏差信息:")
        if self.leveling_system.rotation_angles is not None:
            angles = self.leveling_system.rotation_angles
            self.leveling_text.append(f"旋转角度偏差:")
            self.leveling_text.append(f"  Roll:  {angles[0]:+6.2f}°")
            self.leveling_text.append(f"  Pitch: {angles[1]:+6.2f}°")
            self.leveling_text.append(f"  Yaw:   {angles[2]:+6.2f}°")
        if self.leveling_system.translation_vector is not None:
            t = self.leveling_system.translation_vector
            self.leveling_text.append(f"\n平移量偏差:")
            self.leveling_text.append(f"  X: {t[0]:+6.2f}mm")
            self.leveling_text.append(f"  Y: {t[1]:+6.2f}mm")
            self.leveling_text.append(f"  Z: {t[2]:+6.2f}mm")
        if self.leveling_system.distance_deviation is not None:
            self.leveling_text.append(f"\n平均距离偏差: {self.leveling_system.distance_deviation:.2f}mm")
        if self.leveling_system.rotation_angles is not None:
            angles = self.leveling_system.rotation_angles
            max_angle = max(abs(angles[0]), abs(angles[1]), abs(angles[2]))
            self.leveling_text.append(f"\n{'=' * 30}")
            if max_angle < 0.8:
                self.leveling_text.append("调平状态: 已调平 ✓\n所有角度偏差均小于0.8度")
            else:
                self.leveling_text.append("调平状态: 需要调平 ⚠\n调平建议:")
                if abs(angles[0]) > 0.8:
                    direction = "顺时针" if angles[0] > 0 else "逆时针"
                    self.leveling_text.append(f"• Roll轴需要{direction}调整{abs(angles[0]):.1f}°")
                if abs(angles[1]) > 0.8:
                    direction = "向下" if angles[1] > 0 else "向上"
                    self.leveling_text.append(f"• Pitch轴需要{direction}调整{abs(angles[1]):.1f}°")
                if abs(angles[2]) > 0.8:
                    direction = "顺时针" if angles[2] > 0 else "逆时针"
                    self.leveling_text.append(f"• Yaw轴需要{direction}调整{abs(angles[2]):.1f}°")

    def update_3d_plot(self, points_3d):
        if points_3d is not None:
            if points_3d.ndim == 2 and points_3d.shape[1] == 3:
                points_3d = points_3d[:, np.newaxis, :]
            elif points_3d.ndim == 3 and points_3d.shape[1] != 1:
                points_3d = points_3d[:, :1, :]
        self.ax.clear()
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.set_title('小鼠关键点3D分布 - 调平系统')
        # 删除中轴线设定
        colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown', 'pink']
        if points_3d is not None and len(points_3d) > 0:
            for i, point in enumerate(points_3d):
                x, y, z = point[0, 0], point[0, 1], point[0, 2]
                color = colors[i % len(colors)]
                if i == 0:
                    self.ax.scatter([x], [y], [z], c=color, s=100, marker='o',
                                    label=f'当前-Bregma({x:.1f},{y:.1f},{z:.1f})', alpha=0.8)
                elif i == 1:
                    self.ax.scatter([x], [y], [z], c=color, s=80, marker='o',
                                    label=f'当前-Lambda({x:.1f},{y:.1f},{z:.1f})', alpha=0.8)
                else:
                    point_name = f'P{i - 1}' if i >= 2 else f'P{i}'
                    self.ax.scatter([x], [y], [z], c=color, s=60, marker='o',
                                    label=f'当前-{point_name}', alpha=0.8)
        if self.leveling_system.reference_points is not None:
            ref_points = self.leveling_system.reference_points
            if ref_points.ndim == 2 and ref_points.shape[1] == 3:
                ref_points = ref_points[:, np.newaxis, :]
            elif ref_points.ndim == 3 and ref_points.shape[1] != 1:
                ref_points = ref_points[:, :1, :]
            for i, point in enumerate(ref_points):
                if i < len(ref_points):
                    x, y, z = point[0, 0], point[0, 1], point[0, 2]
                    color = colors[i % len(colors)]
                    if i == 0:
                        self.ax.scatter([x], [y], [z], c='white', s=100, marker='o',
                                        edgecolors=color, linewidth=2,
                                        label=f'参考-Bregma({x:.1f},{y:.1f},{z:.1f})', alpha=0.6)
                    elif i == 1:
                        self.ax.scatter([x], [y], [z], c='white', s=80, marker='o',
                                        edgecolors=color, linewidth=2,
                                        label=f'参考-Lambda({x:.1f},{y:.1f},{z:.1f})', alpha=0.6)
                    else:
                        point_name = f'P{i - 1}' if i >= 2 else f'P{i}'
                        self.ax.scatter([x], [y], [z], c='white', s=60, marker='o',
                                        edgecolors=color, linewidth=2,
                                        label=f'参考-{point_name}', alpha=0.6)
        if points_3d is not None and len(points_3d) > 0:
            all_points = points_3d[:, 0, :]
            if self.leveling_system.reference_points is not None:
                ref_points_coords = self.leveling_system.reference_points[:, 0, :]
                all_points = np.vstack([all_points, ref_points_coords])
            if len(all_points) > 0:
                x_range = max(abs(np.min(all_points[:, 0])), abs(np.max(all_points[:, 0])), 5) + 5
                y_range = max(abs(np.min(all_points[:, 1])), abs(np.max(all_points[:, 1])), 5) + 5
                z_range = max(abs(np.min(all_points[:, 2])), abs(np.max(all_points[:, 2])), 5) + 5
                self.ax.set_xlim([-x_range, x_range])
                self.ax.set_ylim([-y_range, y_range])
                self.ax.set_zlim([-z_range, z_range])
        else:
            range_val = 10
            self.ax.set_xlim([-range_val, range_val])
            self.ax.set_ylim([-range_val, range_val])
            self.ax.set_zlim([-range_val, range_val])
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def update_display(self):
        if self.reconstruction_system and self.reconstruction_system.current_3d_points is not None:
            self.current_points = self.reconstruction_system.current_3d_points
            self.update_coordinates(self.current_points)

    def set_data_source(self, reconstruction_system):
        self.reconstruction_system = reconstruction_system

    def send_motor_command(self, motor_id, direction, steps=5000, speed=30):
        if self.serial_port and self.serial_port.is_open:
            dir_symbol = '+' if direction == 'UP' else '-'
            cmd = f"MA{motor_id}{dir_symbol}{steps:05d}S{speed:04d}"
            print("发送指令：", cmd)
            self.serial_port.write(cmd.encode())
        else:
            print("串口未连接")
            QtWidgets.QMessageBox.critical(self, "串口未连接", "无法发送调平指令，请检查串口连接")

    def start_leveling(self):
        if self.leveling_system.rotation_angles is None:
            QtWidgets.QMessageBox.critical(self, "错误", "未检测到调平角度，无法调平")
            return
        if self.is_leveling:
            QtWidgets.QMessageBox.warning(self, "调平中", "调平已在进行中")
            return
        self.is_leveling = True
        self.leveling_thread = threading.Thread(target=self.leveling_loop)
        self.leveling_thread.daemon = True
        self.leveling_thread.start()
        self.start_leveling_btn.setText("停止调平")
        self.start_leveling_btn.clicked.disconnect()
        self.start_leveling_btn.clicked.connect(self.stop_leveling)
        QtWidgets.QMessageBox.information(self, "调平中", "开始自动调平，系统将循环执行直到达到目标精度或手动停止")

    def stop_leveling(self):
        self.is_leveling = False
        self.start_leveling_btn.setText("开始调平")
        self.start_leveling_btn.clicked.disconnect()
        self.start_leveling_btn.clicked.connect(self.start_leveling)
        QtWidgets.QMessageBox.information(self, "停止调平", "调平已停止")
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")

    def leveling_loop(self):
        max_iterations = 50  # 最大迭代次数
        iteration = 0
        convergence_threshold = 0.8  # 收敛阈值（度）
        while self.is_leveling and iteration < max_iterations:
            try:
                time.sleep(2)  # 等待系统稳定
                if self.leveling_system.rotation_angles is None:
                    print("无法获取当前角度，停止调平")
                    break
                roll, pitch, yaw = self.leveling_system.rotation_angles
                max_angle_error = max(abs(roll), abs(pitch))
                if max_angle_error < convergence_threshold:
                    print(f"调平完成！最大角度误差: {max_angle_error:.2f}°")
                    self.is_leveling = False
                    # 恢复按钮状态
                    QtCore.QMetaObject.invokeMethod(self.start_leveling_btn, "setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "开始调平"))
                    QtCore.QMetaObject.invokeMethod(self.start_leveling_btn.clicked, "disconnect", QtCore.Qt.QueuedConnection)
                    QtCore.QMetaObject.invokeMethod(self.start_leveling_btn.clicked, "connect", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, self.start_leveling))
                    QtWidgets.QMessageBox.information(self, "调平完成", f"调平已完成，最大角度误差: {max_angle_error:.2f}°")
                    break
                target_pitch = 0.0
                target_roll = 0.0
                pitch_error = target_pitch - pitch
                roll_error = target_roll - roll
                pitch_output = self.pid_pitch.compute(pitch_error)
                roll_output = self.pid_roll.compute(roll_error)
                max_output = 2.0  # 最大输出（mm）
                pitch_output = np.clip(pitch_output, -max_output, max_output)
                roll_output = np.clip(roll_output, -max_output, max_output)
                step_length = 1e-4  # 每步长度（mm/step）
                # 新策略：俯仰角由mouth控制，横滚角由left/right对称控制
                mouth_move = -pitch_output
                left_move = -roll_output / 2
                right_move = roll_output / 2
                left_steps = int(abs(left_move / step_length))
                right_steps = int(abs(right_move / step_length))
                mouth_steps = int(abs(mouth_move / step_length))
                left_dir = 'UP' if left_move < 0 else 'DOWN'
                right_dir = 'UP' if right_move > 0 else 'DOWN'
                mouth_dir = 'UP' if mouth_move < 0 else 'DOWN'
                print(f"迭代 {iteration + 1}: Roll={roll:.2f}°, Pitch={pitch:.2f}°")
                print(f"PID输出: Roll={roll_output:.3f}mm, Pitch={pitch_output:.3f}mm")
                print(f"电机步数: Left={left_steps}, Right={right_steps}, Mouth={mouth_steps}")
                self.send_motor_command(1, left_dir, steps=left_steps)
                self.send_motor_command(2, right_dir, steps=right_steps)
                self.send_motor_command(0, mouth_dir, steps=mouth_steps)
                iteration += 1
            except Exception as e:
                print(f"调平过程中出错: {e}")
                break
        self.is_leveling = False
        # 恢复按钮状态
        QtCore.QMetaObject.invokeMethod(self.start_leveling_btn, "setText", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(str, "开始调平"))
        QtCore.QMetaObject.invokeMethod(self.start_leveling_btn.clicked, "disconnect", QtCore.Qt.QueuedConnection)
        QtCore.QMetaObject.invokeMethod(self.start_leveling_btn.clicked, "connect", QtCore.Qt.QueuedConnection, QtCore.Q_ARG(object, self.start_leveling))
        if iteration >= max_iterations:
            QtWidgets.QMessageBox.warning(self, "调平超时", f"达到最大迭代次数({max_iterations})，调平可能未完全收敛")
        elif not self.is_leveling:
            QtWidgets.QMessageBox.information(self, "调平结束", "调平已完成或被手动停止")
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("串口已关闭")

    def show_camera_frame(self, cam2_frame, cam3_frame):
        # BGR转RGB
        cam2_rgb = cv2.cvtColor(cam2_frame, cv2.COLOR_BGR2RGB)
        cam3_rgb = cv2.cvtColor(cam3_frame, cv2.COLOR_BGR2RGB)
        # 转QImage
        h2, w2, ch2 = cam2_rgb.shape
        h3, w3, ch3 = cam3_rgb.shape
        qimg2 = QtGui.QImage(cam2_rgb.data, w2, h2, ch2 * w2, QtGui.QImage.Format_RGB888)
        qimg3 = QtGui.QImage(cam3_rgb.data, w3, h3, ch3 * w3, QtGui.QImage.Format_RGB888)
        # 缩放并显示
        self.cam2_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg2).scaled(self.cam2_label.size(), QtCore.Qt.KeepAspectRatio))
        self.cam3_label.setPixmap(
            QtGui.QPixmap.fromImage(qimg3).scaled(self.cam3_label.size(), QtCore.Qt.KeepAspectRatio))


class IntegratedMainWindow3(QWidget):
    def __init__(self, level_val=None):
        super().__init__()
        self.setWindowTitle("手术界面整合视图")
        self.resize(1800, 900)
        self.level_val = level_val
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.showMaximized()
        # 加载已有的主界面（包含VTK+图谱）
        self.main_window = CoordinateDisplayUI()
        self.main_window.setParent(self)
        layout.addWidget(self.main_window, 4)  # 占比 4
        # 加载步骤窗口
        self.steps = [
            "步骤 1：脑区浏览",
            "步骤 2：路径规划",
            "步骤 3：术前调平",
            "步骤 4：手术执行",
            "步骤 5：注射药物",
            "步骤 6：记录日志"
        ]
        self.steps_widget = SurgeryStepsWidget(self.steps)
        layout.addWidget(self.steps_widget, 1)  # 占比 1

        # 可选：初始化高亮第 0 步
        self.steps_widget.set_current_step(2)

    def on_main_button_clicked(self):
        self.close()


class RealTimeMouse3DReconstruction:
    def __init__(self, camera2_id=1, camera3_id=3):
        # 初始化双目相机参数
        self.dual_projMatr = stereo_camera_caculate_3d_v2.caculate_3d_point_ini()

        # 加载YOLO模型
        self.model = YOLO('best_v3.pt')

        # 定义关键点名称
        self.keypoint_names = [
            'bregma',  # 前囟点 - 索引0，作为原点
            'lambda',  # 后囟点 - 索引1
            'point1',
            'point2',
            'point3',
            'point4',
            'point5'
        ]

        # 初始化相机
        # self.cap2 = cv2.VideoCapture(camera2_id)
        self.cap2 = cv2.VideoCapture(camera2_id)
        self.cap3 = cv2.VideoCapture(camera3_id)

        # 设置相机分辨率
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap3.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 实时数据存储
        self.current_3d_points = None
        self.current_2d_cam2 = None
        self.current_2d_cam3 = None
        self.is_running = True

        # UI界面
        self.coordinate_ui = None
        self.points_3d_buffer = []
        self.filter_window = 5  # 滑动窗口大小，可根据实际调整

    def detect_keypoints_yolo(self, image):
        """使用YOLO模型检测关键点"""
        try:
            results = self.model(image, conf=0.3, verbose=False)

            keypoints = None
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                if len(results[0].keypoints.data) > 0:
                    keypoints = results[0].keypoints.data[0].cpu().numpy()

        except Exception as e:
            print(f"检测过程出错: {e}")
            keypoints = None

        return keypoints

    def convert_to_fontanelle_origin(self, points_3d):
        """将3D坐标转换为以前囟点为原点的坐标系，自动适配维度"""
        if points_3d is None or len(points_3d) < 1:
            return points_3d

        # 自动适配维度
        if points_3d.ndim == 2 and points_3d.shape[1] == 3:
            points_3d = points_3d[:, np.newaxis, :]
        elif points_3d.ndim == 3 and points_3d.shape[1] != 1:
            points_3d = points_3d[:, :1, :]

        fontanelle_point = points_3d[0, 0, :]
        relative_points = points_3d.copy()
        for i in range(len(points_3d)):
            relative_points[i, 0, :] = points_3d[i, 0, :] - fontanelle_point

        return relative_points

    def draw_3d_coordinates_on_image(self, frame, keypoints_2d, points_3d_relative, camera_name):
        """在相机图像上绘制关键点坐标（只显示前囟和后囟坐标，不显示坐标轴）"""
        if keypoints_2d is None or points_3d_relative is None:
            return frame

        # 绘制关键点和坐标
        colors_bgr = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                      (255, 0, 255), (255, 255, 0), (128, 0, 128)]

        # 绘制所有关键点
        if len(keypoints_2d) > 0:
            for i, (name, color) in enumerate(zip(self.keypoint_names, colors_bgr)):
                if i < len(keypoints_2d) and i < len(points_3d_relative):
                    # 2D位置
                    x_2d, y_2d = int(keypoints_2d[i, 0]), int(keypoints_2d[i, 1])

                    # 绘制关键点 - 每个点用不同颜色的单一圆形
                    cv2.circle(frame, (x_2d, y_2d), 12, color, 3)  # 外圈，粗线
                    cv2.circle(frame, (x_2d, y_2d), 6, color, -1)  # 内圈，实心

                    if i == 0:  # bregma - 前囟点
                        cv2.putText(frame, 'Bregma(0,0,0)', (x_2d + 10, y_2d - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    elif name == 'lambda':  # lambda - 后囟点
                        # 获取3D坐标用于显示
                        point_3d = points_3d_relative[i, 0, :]
                        x_3d, y_3d, z_3d = point_3d[0], point_3d[1], point_3d[2]
                        cv2.putText(frame, f'Lambda({x_3d:.1f},{y_3d:.1f},{z_3d:.1f})',
                                    (x_2d + 10, y_2d - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    else:  # 其他关键点 - 只显示点名
                        cv2.putText(frame, f'{name}',
                                    (x_2d + 8, y_2d - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 添加相机信息
        cv2.putText(frame, f'{camera_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, 'Coordinate System: Bregma as Origin', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def process_frames(self):
        """处理相机帧的线程函数"""
        while self.is_running:
            try:
                # 读取双目图像
                ret2, frame2 = self.cap2.read()
                ret3, frame3 = self.cap3.read()

                if not ret2 or not ret3:
                    print("无法读取相机图像")
                    time.sleep(0.1)
                    continue

                # 检测关键点
                keypoints_cam2 = self.detect_keypoints_yolo(frame2)
                keypoints_cam3 = self.detect_keypoints_yolo(frame3)

                if keypoints_cam2 is not None and keypoints_cam3 is not None:
                    if len(keypoints_cam2) >= 7 and len(keypoints_cam3) >= 7:
                        # 提取2D坐标
                        points_2d_cam2 = keypoints_cam2[:7, :2]
                        points_2d_cam3 = keypoints_cam3[:7, :2]

                        # 保存2D坐标
                        self.current_2d_cam2 = points_2d_cam2
                        self.current_2d_cam3 = points_2d_cam3

                        # 计算3D坐标
                        points_3d = stereo_camera_caculate_3d_v2.caculate_3d_point(
                            points_2d_cam2, points_2d_cam3, self.dual_projMatr)

                        # 转换为前囟坐标系
                        points_3d_relative = self.convert_to_fontanelle_origin(points_3d)

                        # 更新当前3D点数据（加入滤波）
                        if points_3d_relative is not None:
                            self.points_3d_buffer.append(points_3d_relative)
                            if len(self.points_3d_buffer) > self.filter_window:
                                self.points_3d_buffer.pop(0)
                            # 滤波：对最近N帧取均值
                            filtered_points = np.mean(self.points_3d_buffer, axis=0)
                            self.current_3d_points = filtered_points
                        else:
                            self.current_3d_points = points_3d_relative

                        #print("成功计算3D坐标")

                # 在图像上绘制3D坐标系和坐标信息
                frame2_with_coords = self.draw_3d_coordinates_on_image(
                    frame2.copy(), self.current_2d_cam2, self.current_3d_points, "Camera 2")
                frame3_with_coords = self.draw_3d_coordinates_on_image(
                    frame3.copy(), self.current_2d_cam3, self.current_3d_points, "Camera 3")

                if self.coordinate_ui is not None:
                    self.coordinate_ui.main_window.show_camera_frame(frame2_with_coords, frame3_with_coords)
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.is_running = False
                    break

            except Exception as e:
                print(f"帧处理错误: {e}")

            time.sleep(0.03)  # 控制帧率约30fps

    def run_real_time(self):
        """运行实时3D重建"""
        print("启动实时3D重建...")
        print("按 'q' 键退出")

        # 启动坐标显示UI（在新线程中）
        def start_ui():
            app = QtWidgets.QApplication(sys.argv)
            self.coordinate_ui = IntegratedMainWindow3()
            self.coordinate_ui.main_window.set_data_source(self)
            self.coordinate_ui.show()
            app.exec_()

        ui_thread = threading.Thread(target=start_ui)
        ui_thread.daemon = True
        ui_thread.start()

        # 等待UI启动
        time.sleep(1)

        # 直接在主线程中处理相机
        self.process_frames()

        # 清理资源
        self.cleanup()

    def cleanup(self):
        """清理资源"""
        print("清理资源...")
        self.is_running = False

        if hasattr(self, 'cap2') and self.cap2.isOpened():
            self.cap2.release()
        if hasattr(self, 'cap3') and self.cap3.isOpened():
            self.cap3.release()

        cv2.destroyAllWindows()

        # 关闭UI
        if self.coordinate_ui is not None:
            try:
                self.coordinate_ui.root.quit()
                self.coordinate_ui.root.destroy()
            except:
                pass


def STA():
    # 创建实时重建系统
    # real_time_reconstruction = RealTimeMouse3DReconstruction(camera2_id=0, camera3_id=3)
    real_time_reconstruction = RealTimeMouse3DReconstruction(camera2_id=0, camera3_id=1)

    # 运行实时重建
    real_time_reconstruction.run_real_time()


if __name__ == "__main__":
    STA()