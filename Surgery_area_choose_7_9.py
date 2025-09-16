import sys
import nrrd
import os
import numpy as np
import json
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPalette

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from allensdk.core.reference_space_cache import ReferenceSpaceCache

class SliceCanvas(FigureCanvasQTAgg):
    def __init__(self, orientation, brain_data, on_click_callback, on_motion_callback, initial_index=None):
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        super().__init__(self.fig)
        self.orientation = orientation
        self.brain_data = brain_data
        self.slice_index = (
            initial_index
            if initial_index is not None
            else brain_data.shape[orientation] // 2
        )
        self.on_click_callback = on_click_callback
        self.on_motion_callback = on_motion_callback

        self.cross_x = None
        self.cross_y = None

        self.update_slice()
        self.mpl_connect("button_press_event", self.mouse_clicked)
        self.mpl_connect("motion_notify_event", self.mouse_moved)

        self.fig.patch.set_facecolor('#222222')
        self.ax.set_facecolor('#222222')
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    def update_slice(self):
        if self.orientation == 0:
            img = self.brain_data[self.slice_index, :, :]
        elif self.orientation == 1:
            img = self.brain_data[:, self.slice_index, :]
        else:
            img = self.brain_data[:, :, self.slice_index]

        self.ax.clear()
        self.ax.imshow(img.T, cmap="gray", origin="lower")
        self.ax.set_title(["Sagittal", "Coronal", "Axial"][self.orientation],
                          color='white', fontsize=12, fontweight='bold')

        self.ax.set_xticks(np.linspace(0, img.shape[0], 5))
        self.ax.set_yticks(np.linspace(0, img.shape[1], 5))
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        self.ax.tick_params(colors='white', labelsize=10)

        if self.cross_x is not None and self.cross_y is not None:
            self.ax.axhline(self.cross_y, color='red', linestyle='--', linewidth=1)
            self.ax.axvline(self.cross_x, color='red', linestyle='--', linewidth=1)

        self.draw()

    def set_crosshair(self, x, y):
        self.cross_x = x
        self.cross_y = y
        self.update_slice()

    def mouse_clicked(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        coords = [0, 0, 0]
        if self.orientation == 0:
            coords = [self.slice_index, x, y]
        elif self.orientation == 1:
            coords = [x, self.slice_index, y]
        else:
            coords = [x, y, self.slice_index]
        self.on_click_callback(coords, self.orientation)

    def mouse_moved(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self.orientation == 0:
            self.on_motion_callback(int(event.xdata), int(event.ydata))


class BrainApp(QWidget):
    def __init__(self, level_val=None):
        super().__init__()
        self.setWindowTitle("Allen 小鼠脑图谱浏览器")
        self.annotation, _ = nrrd.read("data/average_template_10.nrrd")

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#121212"))
        palette.setColor(QPalette.WindowText, Qt.black)
        self.setPalette(palette)

        CACHE_DIR = "./allen_ccf_cache"
        os.makedirs(CACHE_DIR, exist_ok=True)
        rspc = ReferenceSpaceCache(
            resolution=25,
            reference_space_key='annotation/ccf_2017',
            manifest=os.path.join(CACHE_DIR, 'manifest.json')
        )
        self.tree = rspc.get_structure_tree()

        self.target_coords = None
        self.preview_coords = None
        self.entry_coords = None
        self.preview_line = None
        self.target_struct_name = None  # 新增：保存目标点脑区名称

        self.name_label = QLabel("点击 Sagittal 面选择注射目标")
        self.name_label.setStyleSheet("color: white; background-color: #333333; padding: 6px; border-radius: 6px;")
        font_label = QFont("Arial", 14, QFont.Bold)
        self.name_label.setFont(font_label)
        self.name_label.setAlignment(Qt.AlignCenter)
        self.coord_distance_label = QLabel("距离前囟点的坐标：(-, -, -) mm")
        self.coord_distance_label.setStyleSheet("color: white; background-color: #333333; padding: 6px; border-radius: 6px;")
        self.coord_distance_label.setFont(font_label)

        self.layout = QVBoxLayout(self)

        self.slice_canvases = []
        canvas_layout = QHBoxLayout()
        for ori in range(3):
            canvas = SliceCanvas(ori, self.annotation, self.on_voxel_clicked, self.on_mouse_moved, level_val)
            self.slice_canvases.append(canvas)
            canvas_layout.addWidget(canvas)
        self.layout.addLayout(canvas_layout)

        self.confirm_button = QPushButton("确认注射脑区")
        self.confirm_button.clicked.connect(self.confirm_and_save)
        self.back_button = QPushButton("返回上一步")
        self.back_button.clicked.connect(self.go_back)
        self.next_button = QPushButton("开始调平")
        self.next_button.clicked.connect(self.go_next)
        self.layout.addWidget(self.confirm_button)
        self.layout.addWidget(self.back_button)
        self.layout.addWidget(self.next_button)

        if level_val is not None:
            init_sagittal = min(max(int(level_val), 0), self.annotation.shape[0] - 1)
        else:
            init_sagittal = self.annotation.shape[0] // 2

        slider_layout = QHBoxLayout()
        self.sliders = []
        self.slider_value_labels = []
        font_slider_label = QFont("Arial", 10, QFont.Bold)

        for i, ori in enumerate(["Sagittal", "Coronal", "Axial"]):
            vbox = QVBoxLayout()
            label = QLabel(ori)
            label.setFont(font_slider_label)
            label.setStyleSheet("color: white;")

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self.annotation.shape[i] - 1)

            if i == 0:
                slider.setValue(init_sagittal)
                self.slice_canvases[0].slice_index = init_sagittal
            else:
                slider.setValue(self.annotation.shape[i] // 2)

            slider.valueChanged.connect(lambda val, i=i: self.update_slice(i, val))

            value_label = QLabel(str(slider.value()))
            value_label.setFont(font_slider_label)
            value_label.setStyleSheet("color: #AAAAAA;")
            value_label.setAlignment(Qt.AlignCenter)

            self.sliders.append(slider)
            self.slider_value_labels.append(value_label)

            vbox.addWidget(label)
            vbox.addWidget(slider)
            vbox.addWidget(value_label)
            slider_layout.addLayout(vbox)

        self.layout.addLayout(slider_layout)
        self.layout.addWidget(self.name_label)
        self.layout.addWidget(self.coord_distance_label)

    def go_back(self):
        from step1_plus_7_3 import IntegratedMainWindow as PreviousWindow
        try:
            self.previous_window = PreviousWindow()
            self.previous_window.show()
            self.close()
        except Exception as e:
            print("窗口打开失败：", e)

    def go_next(self):
        from Auto_adjust import STA
        print("导入成功")
        try:
            print("11")
            STA()
            self.close()
        except Exception as e:
            print("窗口打开失败：", e)

    def update_slice(self, orientation, val):
        self.slice_canvases[orientation].slice_index = val
        self.slice_canvases[orientation].update_slice()
        self.slider_value_labels[orientation].setText(str(val))

    def update_coordinate_distance(self, x, y, z):
        spacing = 0.010
        bregma_voxel = np.array([570, 44, 570])
        delta = (np.array([x, y, z]) - bregma_voxel) * spacing
        delta_mm = tuple(np.round(delta, 2))
        self.coord_distance_label.setText(f"距离前囟点坐标：({delta_mm[0]} mm, {delta_mm[1]} mm, {delta_mm[2]} mm)")

    def on_voxel_clicked(self, coords, sender_orientation):
        if sender_orientation != 0:
            return

        x, y, z = coords
        # 每次点击都直接设置目标点
        self.target_coords = [x, y, z]
        struct_id = int(self.annotation[x, y, z])
        struct = self.tree.get_structures_by_id([struct_id])
        name = struct[0]['name'] if struct else '未知结构'
        self.target_struct_name = name
        self.name_label.setText(f"注射目标点: ({x}, {y}, {z}) → {name}")

        # 清除之前的入口点
        self.entry_coords = None

        # 更新滑块和 crosshair
        self.sliders[1].setValue(y)
        self.sliders[2].setValue(z)
        self.slice_canvases[1].set_crosshair(x, z)
        self.slice_canvases[2].set_crosshair(x, y)
        self.update_coordinate_distance(x, y, z)

        # 清除之前的路径
        self.slice_canvases[0].ax.lines.clear()
        self.slice_canvases[0].draw()

    def on_mouse_moved(self, x, y):
        # 删除跟随鼠标移动的黄线预览功能
        pass

    def draw_preview_path(self):
        # 删除预览路径绘制功能
        pass

    def draw_injection_path(self):
        sag_ax = self.slice_canvases[0].ax
        sag_ax.lines.clear()
        sag_ax.plot(
            [self.target_coords[1], self.entry_coords[1]],
            [self.target_coords[2], self.entry_coords[2]],
            color='lime', linestyle='-', linewidth=2, marker='o'
        )
        self.slice_canvases[0].draw()

    def confirm_and_save(self):
        if self.target_coords is None or self.entry_coords is None:
            self.name_label.setText("请先选择目标点和入口点")
            return

        spacing = 0.010
        bregma_voxel = np.array([570, 44, 570])

        def mm_coords(voxel):
            delta = (np.array(voxel) - bregma_voxel) * spacing
            return list(np.round(delta, 3))

        data = {
            "target_voxel": self.target_coords,
            "entry_voxel": self.entry_coords,
            "target_mm": mm_coords(self.target_coords),
            "entry_mm": mm_coords(self.entry_coords),
            "target_struct_name": self.target_struct_name  # 新增：保存脑区名称
        }
        with open("injection_path.json", "w") as f:
            json.dump(data, f, indent=4)

        self.name_label.setText("注射路径已保存 ✅")
        print("注射路径已保存到 injection_path.json")
        self.target_coords = None
        self.entry_coords = None
        self.preview_coords = None
        self.slice_canvases[0].ax.lines.clear()
        self.slice_canvases[0].draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BrainApp(level_val=146)
    win.resize(1300, 900)
    win.show()
    sys.exit(app.exec_())