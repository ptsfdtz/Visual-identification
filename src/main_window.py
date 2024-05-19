from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import cv2
from face_detection import FaceDetector

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.face_detector = FaceDetector()
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(10)
        self.detecting = False

    def initUI(self):
        self.setGeometry(100, 100, 640, 480)
        self.setWindowTitle('人脸识别')

        self.label = QLabel(self)
        self.label.resize(640, 480)

        self.detectButton = QPushButton('识别人脸', self)
        self.detectButton.clicked.connect(self.toggleDetection)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.detectButton)

        self.setLayout(vbox)

    def toggleDetection(self):
        self.detecting = not self.detecting
        if self.detecting:
            self.detectButton.setText('停止')
        else:
            self.detectButton.setText('识别人脸')

    def updateFrame(self):
        ret, frame = self.face_detector.get_frame(self.detecting)
        if ret:
            self.displayFrame(frame)

    def displayFrame(self, frame):
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, aspectRatioMode=Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap.fromImage(p))
