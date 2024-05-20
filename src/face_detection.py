import cv2
import sqlite3
import numpy as np

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = cv2.VideoCapture(0)
        self.conn = sqlite3.connect('faces.db')
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            image BLOB NOT NULL)''')
        self.conn.commit()

    def insert_face(self, name, image_data):
        success, encoded_image = cv2.imencode('.jpg', image_data)
        if success:
            self.cursor.execute("INSERT INTO faces (name, image) VALUES (?, ?)", 
                                (name, encoded_image.tobytes()))
            self.conn.commit()

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def save_face(self, name, frame):
        faces = self.detect_face(frame)
        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]
            self.insert_face(name, face_image)

    def load_known_faces(self):
        self.cursor.execute("SELECT name, image FROM faces")
        known_faces = {}
        for row in self.cursor.fetchall():
            name = row[0]
            image = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_COLOR)
            known_faces[name] = image
        return known_faces

    def match_face(self, face_image):
        known_faces = self.load_known_faces()
        for name, known_face_image in known_faces.items():
            if self.is_match(face_image, known_face_image):
                return name
        return None

    def is_match(self, face_image, known_face_image):
        # 简单的模板匹配方法
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        known_face_gray = cv2.cvtColor(known_face_image, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(face_gray, known_face_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        return max_val > 0.6  # 可调整阈值

    def __del__(self):
        self.cap.release()
        self.conn.close()
