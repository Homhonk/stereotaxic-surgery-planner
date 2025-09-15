import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout
from Brain_level_choose import MainWindow  # 替换为你的主界面类所在文件
from step_shower import SurgeryStepsWidget


class IntegratedMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手术界面整合视图")
        self.resize(1800, 900)

        layout = QHBoxLayout()
        self.setLayout(layout)
        self.showMaximized()
        # 加载已有的主界面（包含VTK+图谱）
        self.main_window = MainWindow()
        self.main_window.setParent(self)
        layout.addWidget(self.main_window, 4)  # 占比 4
        self.main_window.start_layer_btn.clicked.connect(self.on_main_button_clicked)
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
        self.steps_widget.set_current_step(0)

    def on_main_button_clicked(self):
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = IntegratedMainWindow()
    window.show()
    sys.exit(app.exec_())
