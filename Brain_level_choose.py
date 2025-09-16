import sys
import os
print("PATH:", os.environ['PATH'])
from step2_plus_7_3 import IntegratedMainWindow2 as pro
import cv2
import vtk
import nrrd
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QCheckBox, QPushButton, QSlider, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTK模型 + NRRD切面/图谱")
        self.setGeometry(100, 100, 1600, 900)

        app = QApplication.instance()  # 取到 QApplication 单例
        if app:
            font = app.font()
            font.setFamily("Microsoft YaHei")  # 微软雅黑字体
            font.setPointSize(11)
            app.setFont(font)

            dark_stylesheet = """
                    QWidget {
                        background-color: #2b2b2b;
                        color: #ffffff;
                    }
                    QPushButton {
                        background-color: #3c3f41;
                        border: 1px solid #5c5c5c;
                        padding: 5px;
                    }
                    QPushButton:hover {
                        background-color: #484a4c;
                    }
                    QSlider::groove:vertical {
                        background: #3c3f41;
                        width: 6px;
                        border-radius: 3px;
                    }
                    QSlider::handle:vertical {
                        background: #9c9c9c;
                        height: 14px;
                        margin: 0 -2px;
                        border-radius: 3px;
                    }
                    QCheckBox {
                        spacing: 5px;
                    }
                    QCheckBox::indicator {
                        width: 14px;
                        height: 14px;
                    }
                    QLabel {
                        font-size: 12px;
                    }
                    """
            app.setStyleSheet(dark_stylesheet)
        else:
            print("QApplication实例未初始化！")
        # 主容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # -------------------- 上：模型显示窗口 --------------------
        self.vtk_widget_model = QVTKRenderWindowInteractor()
        #self.vtk_widget_model.setEnabled(False)

        layout.addWidget(self.vtk_widget_model, 5)
        #创建一个辅助选层的参考平面
        self.renderer_model = vtk.vtkRenderer()
        self.vtk_widget_model.GetRenderWindow().AddRenderer(self.renderer_model)
        self.plane = vtk.vtkCubeSource()
        self.plane.SetXLength(15)  # 宽度
        self.plane.SetYLength(20)  # 高度
        self.plane.SetZLength(0.3)  # 厚度（让它不是无厚度的平面）
        self.plane.SetCenter(-10, -10, -30)  # 根据滑动条值设置Z位置
        self.plane.Update()

        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputConnection(self.plane.GetOutputPort())

        self.plane_actor = vtk.vtkActor()
        self.plane_actor.SetMapper(plane_mapper)
        self.plane_actor.GetProperty().SetColor(1, 0, 0)  # 红色平面
        self.plane_actor.GetProperty().SetOpacity(0.5)  # 透明度
        self.plane_actor.GetProperty().SetBackfaceCulling(False)
        self.plane_actor.GetProperty().SetFrontfaceCulling(False)

        self.renderer_model.AddActor(self.plane_actor)
        self.model_checkboxes = []
        checkbox_layout = QHBoxLayout()
        for name in ["ms_bone", "ms_brain", "ms_skin", "ms_lung", "frame"]:
            cb = QCheckBox(name)
            cb.stateChanged.connect(self.toggle_model)
            self.model_checkboxes.append(cb)
            checkbox_layout.addWidget(cb)
        layout.addLayout(checkbox_layout)

        button_layout = QHBoxLayout()
        self.start_layer_btn = QPushButton("开始选区")
        self.start_layer_btn.clicked.connect(self.choose_level)
        button_layout.addWidget(self.start_layer_btn)
        layout.addLayout(button_layout)

        self.models = {}  # 存放模型 actor

        # -------------------- 下：nrrd切面 + 图谱显示 --------------------
        bottom_layout = QHBoxLayout()
        layout.addLayout(bottom_layout, 4)

        self.vtk_widget_nrrd = QVTKRenderWindowInteractor()
        bottom_layout.addWidget(self.vtk_widget_nrrd, 6)

        self.renderer_nrrd = vtk.vtkRenderer()
        self.vtk_widget_nrrd.GetRenderWindow().AddRenderer(self.renderer_nrrd)

        self.slider = QSlider(Qt.Vertical)
        self.slider.setMinimum(1)
        self.slider.setMaximum(94)
        self.slider.valueChanged.connect(self.update_slice)
        bottom_layout.addWidget(self.slider, 1)

        self.position_label = QLabel("当前层面相对前囟点偏移：0")
        bottom_layout.addWidget(self.position_label,1)

        self.atlas_label = QLabel()
        self.atlas_label.setFixedSize(640, 540)
        self.atlas_label.setStyleSheet("background-color: black")
        self.atlas_label.setAlignment(Qt.AlignCenter)
        self.atlas_label.mouseDoubleClickEvent = self.show_full_atlas_image  # 👈 绑定双击事件
        bottom_layout.addWidget(self.atlas_label, 1)

        self.atlas_folder = "./brain_map"  # 存放1.png到100.png的文件夹
        self.front_point_index = 31  # 假设前囟点对应滑动条值605（用户可调整）
        self.mm_range = 4.25 + 7.47
        self.mm_per_index = 0.12  # 共99间隔，对应100张图谱
        self.set_fixed_camera()
        self.load_nrrd_slice("./data/average_template_10.nrrd")

    def toggle_model(self):
        for cb in self.model_checkboxes:
            name = cb.text()
            path = f"./models/{name}.vtk"
            if cb.isChecked() and name not in self.models:
                reader = vtk.vtkPolyDataReader()
                reader.SetFileName(path)
                reader.Update()
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(reader.GetOutput())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetSpecular(.3)
                actor.GetProperty().SetSpecularPower(20)
                actor.GetProperty().SetOpacity(0.5)
                self.renderer_model.AddActor(actor)
                self.models[name] = actor

                if name == "frame":
                    self.set_camera_to_frame(reader.GetOutput())

            elif not cb.isChecked() and name in self.models:
                self.renderer_model.RemoveActor(self.models[name])
                del self.models[name]
        self.vtk_widget_model.GetRenderWindow().Render()


    def choose_level(self):
        try:
            #self.choose_win = pro(level_val=(self.slider.value() - 1) * 12.5 + 146)
            self.choose_win = pro(level_val=int((self.slider.value() - 1) * 12.5 + 146))
            self.choose_win.show()
        except Exception as e:
            print("窗口打开失败：", e)


    def set_camera_to_frame(self, frame_polydata):
        camera = self.renderer_model.GetActiveCamera()
        bounds = frame_polydata.GetBounds()
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
        camera.SetFocalPoint(center)
        camera.SetPosition(center[0], center[1], center[2] + 100)
        camera.SetViewUp(0, 1, 0)
        self.renderer_model.ResetCameraClippingRange()

    def load_nrrd_slice(self, nrrd_path):
        data, header = nrrd.read(nrrd_path)
        self.slice_data = data
        self.image_import = vtk.vtkImageImport()
        self.actor_nrrd = vtk.vtkImageActor()
        self.renderer_nrrd.AddActor(self.actor_nrrd)
        self.update_slice()

    def update_slice(self):
        slider_val = self.slider.value()
        print(slider_val)
        if not hasattr(self, "slice_data"):
            return

        # 映射至 nrrd 层索引
        real_idx = int((slider_val - 1)*12.5+146)
        slice_img = self.slice_data[real_idx, :, :].copy(order='C')
        slice_img = ((slice_img - slice_img.min()) / (np.ptp(slice_img)) * 255).astype(np.uint8)
        slice_img = np.flipud(slice_img)  # 上下翻转
        h, w = slice_img.shape
        self.image_import.CopyImportVoidPointer(slice_img.tobytes(), len(slice_img.tobytes()))
        self.image_import.SetDataScalarTypeToUnsignedChar()
        self.image_import.SetNumberOfScalarComponents(1)
        self.image_import.SetWholeExtent(0, w - 1, 0, h - 1, 0, 0)
        self.image_import.SetDataExtentToWholeExtent()
        self.image_import.Update()
        self.actor_nrrd.GetMapper().SetInputConnection(self.image_import.GetOutputPort())
        self.renderer_nrrd.ResetCamera()
        self.vtk_widget_nrrd.GetRenderWindow().Render()
        #
        self.update_plane_position(slider_val)
        # 偏移值更新（单位：mm）
        offset = round(4.25-(0.124*(slider_val - 1)), 2)
        self.position_label.setText(f"当前层面相对前囟点偏移：{offset} mm")

        # 图谱图像更新
        atlas_idx = slider_val # 1~100
        img_path = os.path.join(self.atlas_folder, f"{atlas_idx}.png")
        if os.path.exists(img_path):
            pix = QPixmap(img_path).scaled(self.atlas_label.width(), self.atlas_label.height(), Qt.KeepAspectRatio)
            self.atlas_label.setPixmap(pix)
        else:
            self.atlas_label.setText("无图谱")


    def show_full_atlas_image(self, event):
        print("ENTER")
        slider_val = self.slider.value()
        img_path = "brain_map/"+f"{slider_val}.png"
        print(img_path)
        map = cv2.imread(img_path)
        cv2.imshow(map)
        print(img_path)

    def update_plane_position(self, slider_val):
        z_pos = self.slice_to_world_z(slider_val)  # 将滑动条值映射到世界坐标
        print(z_pos)
        self.plane.SetCenter(0, 0, z_pos)
        self.plane.Update()
        self.vtk_widget_model.GetRenderWindow().Render()

    def slice_to_world_z(self, slice_index):
        spacing = 0.12  # 每层0.1mm，例如
        origin = -30  # 前囟点前面4.25mm为起点

        return origin + slice_index * spacing

    def set_fixed_camera(self):
        camera = self.renderer_model.GetActiveCamera()
        camera.SetPosition(50, 0, 0)       # 相机位置
        camera.SetFocalPoint(0, 0, 0)       # 焦点（模型中心）
        #camera.SetViewUp(0, 0, 1)           # 向上方向
        camera.SetViewUp(0, 1, 0)  # 向上方向
        self.renderer_model.ResetCameraClippingRange()
        self.vtk_widget_model.GetRenderWindow().Render()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())