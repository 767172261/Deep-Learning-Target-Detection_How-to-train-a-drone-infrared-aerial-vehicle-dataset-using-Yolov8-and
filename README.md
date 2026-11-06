深度学习目标检测中_如何使用Yolov8训练无人机红外航拍人车数据集,并构建一个基于深度学习的模型来识别 无人机红外检测数据![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/090e6df08c42416d90ba34284fb002d7.png)
并构建一个基于深度学习的模型来识别 无人机红外检测数据
目标检测 YOLO格式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08ff971cd2d142a0bc591c60a3d7ffeb.png)
并构建一个基于深度学习的模型来识别 无人机红外检测数据
目标检测 YOLO格式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0970e62d021c4904a73b5043dd15bde3.png)

## 以下文字及代码仅供参考：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d042602744c4cce96ebd7fb85bb86c3.png)

使用YOLOv8训练一个无人机红外航拍人车检测模型，来识别检测红外人车检测

### 1. 环境部署

确保安装了所有必要的依赖库：

```bash
pip install -r requirements.txt
```

`requirements.txt`内容示例：

```
torch>=1.9.0
opencv-python
timm
psutil
PyQt6
numpy
matplotlib
ultralytics
```

### 2. 数据预处理

#### 数据集准备

对于红外图像数据集，你需要确保你的数据集按照YOLO格式组织，并包含以下内容：

- 图像文件夹 (`images/`)
- 标签文件夹 (`labels/`)
- 数据集配置文件 (`dataset.yaml`)

每个图像对应一个同名的`.txt`文件作为标注文件，其中包含物体的位置信息（边界框坐标）和类别ID。标签文件应包括行人（person）和车辆（car）两种类别。

```yaml
# dataset.yaml
train: ./dataset/images/train
val: ./dataset/images/valid

nc: 2  # Number of classes (person, car)
names: ['person', 'car']
```

#### 划分数据集

编写脚本将数据集划分为训练集和验证集，并为每个集合创建相应的标签文件。

```python
import random
import shutil
import os

def split_dataset(src_dir, dst_train_dir, dst_val_dir, val_ratio=0.2):
    images = [f for f in os.listdir(os.path.join(src_dir, 'images')) if f.endswith(('png', 'jpg', 'jpeg'))]
    labels = [f for f in os.listdir(os.path.join(src_dir, 'labels')) if f.endswith('txt')]

    random.shuffle(images)
    split_idx = int(len(images) * (1 - val_ratio))

    for i, img in enumerate(images):
        label = img.replace('.jpg', '.txt').replace('.png', '.txt')
        if i < split_idx:
            shutil.copy(os.path.join(src_dir, 'images', img), os.path.join(dst_train_dir, 'images', img))
            shutil.copy(os.path.join(src_dir, 'labels', label), os.path.join(dst_train_dir, 'labels', label))
        else:
            shutil.copy(os.path.join(src_dir, 'images', img), os.path.join(dst_val_dir, 'images', img))
            shutil.copy(os.path.join(src_dir, 'labels', label), os.path.join(dst_val_dir, 'labels', label))

# 示例调用
split_dataset('path/to/dataset', 'path/to/train', 'path/to/val')
```

### 3. 模型定义

假设你已经有了YOLOv8的安装和配置，这里直接定义检测器类。

```python
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8_best.pt', conf_thres=0.5, iou_thres=0.45):
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def detect(self, image, conf_thres=None, iou_thres=None):
        results = self.model(image, conf=conf_thres or self.conf_thres, iou=iou_thres or self.iou_thres)
        return results[0].boxes.cpu().numpy()

    def draw_boxes(self, image, boxes, labels=None):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if labels:
                label = labels[int(box.cls)]
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return image
```

### 4. 训练模型

编写训练脚本来启动训练过程。考虑到红外图像可能有不同的特性，可以尝试调整一些超参数以优化性能。

```python
from yolov8_detector import YOLOv8Detector

def train_model():
    detector = YOLOv8Detector(model_path='models/yolov8n.yaml')  # 使用预定义配置或自定义模型
    results = detector.model.train(data='dataset.yaml', epochs=100, imgsz=640, batch=16)

if __name__ == '__main__':
    train_model()
```

### 5. 评估模型

编写评估脚本来测试模型性能，并计算mAP等指标。

```python
from yolov8_detector import YOLOv8Detector

def evaluate_model():
    detector = YOLOv8Detector(model_path='runs/detect/train/weights/best.pt')  # 加载最佳模型权重
    results = detector.model.val(data='dataset.yaml')
    print(results)

if __name__ == '__main__':
    evaluate_model()
```

### 6. 结果分析与可视化

编写脚本以分析和可视化检测结果。

```python
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_detection_results(image_paths, detection_results):
    fig, axes = plt.subplots(len(image_paths), 1, figsize=(10, len(image_paths) * 5))
    if len(image_paths) == 1:
        axes = [axes]

    for ax, image_path, result in zip(axes, image_paths, detection_results):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for det in result:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        ax.imshow(image)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 示例调用
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
detection_results = [detector.detect(cv2.imread(path)) for path in image_paths]
plot_detection_results(image_paths, detection_results)
```

### 7. 集成与部署

使用PyQt6构建图形用户界面，并集成上述功能模块。

#### PyQt6 GUI (`gui.py`)

```python
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QLineEdit, QComboBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer
import cv2
import numpy as np
from yolov8_detector import YOLOv8Detector  # 假设这是你的YOLOv8检测器类

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("无人机红外航拍人车检测系统")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout(self.central_widget)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(['Yolov8', 'Yolov8-Ghost', 'Yolov8-CCFM'])
        self.model_selector.currentIndexChanged.connect(self.change_model)
        self.layout.addWidget(self.model_selector)

        self.load_video_button = QPushButton('选择视频', self)
        self.load_video_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.load_video_button)

        self.start_camera_button = QPushButton('打开摄像头', self)
        self.start_camera_button.clicked.connect(self.start_camera)
        self.layout.addWidget(self.start_camera_button)

        self.iou_input = QLineEdit(self)
        self.iou_input.setPlaceholderText('输入IOU阈值')
        self.layout.addWidget(self.iou_input)

        self.conf_input = QLineEdit(self)
        self.conf_input.setPlaceholderText('输入置信度阈值')
        self.layout.addWidget(self.conf_input)

        self.save_button = QPushButton('保存结果', self)
        self.save_button.clicked.connect(self.save_result)
        self.layout.addWidget(self.save_button)

        self.models = {
            'Yolov8': 'models/yolov8_best.pt',
            'Yolov8-Ghost': 'models/yolov8_ghost_best.pt',
            'Yolov8-CCFM': 'models/yolov8_ccfm_best.pt'
        }
        self.detector = YOLOv8Detector(model_path=self.models['Yolov8'])  # 默认加载第一个模型

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.video_capture = None
        self.current_frame = None

    def change_model(self, index):
        selected_model = self.model_selector.itemText(index)
        self.detector = YOLOv8Detector(model_path=self.models[selected_model])

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.start_video(file_name)

    def start_camera(self):
        self.start_video(0)

    def start_video(self, source):
        self.video_capture = cv2.VideoCapture(source)
        self.timer.start(30)  # 设置定时器间隔，单位为毫秒

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            iou_thres = float(self.iou_input.text()) if self.iou_input.text() else 0.45
            conf_thres = float(self.conf_input.text()) if self.conf_input.text() else 0.5
            results = self.detector.detect(frame, conf_thres=conf_thres, iou_thres=iou_thres)
            frame_with_boxes = self.detector.draw_boxes(frame, results)

            rgb_image = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = convert_to_Qt_format.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(QPixmap.fromImage(p))
            self.current_frame = frame_with_boxes
        else:
            self.timer.stop()

    def save_result(self):
        if self.current_frame is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "Images (*.png *.xpm *.jpg *.bmp *.gif)")
            if file_name:
                cv2.imwrite(file_name, self.current_frame)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```



那么这样的化。通过以上步骤，可以构建一个完整的无人机红外航拍人车检测系统。
