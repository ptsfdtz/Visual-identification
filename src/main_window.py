from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from face_detection import FaceDetector
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.face_detector = FaceDetector()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(10)
        self.detecting = False
        self.detected_faces = []

    def initUI(self):
        self.setGeometry(100, 100, 640, 480)
        self.setWindowTitle('Face Detection and Recognition App')

        self.label = QLabel(self)
        self.label.resize(640, 480)

        self.nameInput = QLineEdit(self)
        self.nameInput.setPlaceholderText('Enter name')

        self.detectButton = QPushButton('Detect Face', self)
        self.detectButton.clicked.connect(self.startDetection)

        self.saveButton = QPushButton('Save Face', self)
        self.saveButton.clicked.connect(self.saveFace)
        self.saveButton.setEnabled(False)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.nameInput)
        vbox.addWidget(self.detectButton)
        vbox.addWidget(self.saveButton)

        self.setLayout(vbox)

    def startDetection(self):
        self.detecting = True
        self.detectButton.setEnabled(False)
        self.saveButton.setEnabled(True)
        self.nameInput.setEnabled(True)

    def saveFace(self):
        name = self.nameInput.text()
        if name and self.detected_faces:
            for face in self.detected_faces:
                self.face_detector.save_face(name, face)
            self.nameInput.clear()
            self.detecting = False
            self.detectButton.setEnabled(True)
            self.saveButton.setEnabled(False)

    def updateFrame(self):
        ret, frame = self.face_detector.get_frame()
        if ret:
            if self.detecting:
                self.detectAndDisplayFace(frame)
            else:
                self.displayFrame(frame)

    def displayFrame(self, frame):
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.label.setPixmap(QPixmap.fromImage(p))

    def detectAndDisplayFace(self, frame):
        faces = self.face_detector.detect_face(frame)
        self.detected_faces = []
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            name = self.face_detector.match_face(face_image)
            if name:
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            self.detected_faces.append(face_image)
        self.displayFrame(frame)

    def closeEvent(self, event):
        del self.face_detector
