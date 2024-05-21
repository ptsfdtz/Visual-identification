import cv2
import os
from tqdm import tqdm

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

face_id = input('\n Enter user id (integer): ')

facedata_dir = f'Facedata/User_{face_id}'
os.makedirs(facedata_dir, exist_ok=True)

print('\n Initializing face capture. Look at the camera and wait ...')

count = 0
total_samples = 2000 #样本数量自行调整

with tqdm(total=total_samples) as pbar:
    while True:
        success, img = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))
            count += 1
            cv2.imwrite(os.path.join(facedata_dir, f'User_{face_id}_{count}.jpg'), gray[y: y + h, x: x + w])
            pbar.update(1)
            cv2.imshow('image', img)
        k = cv2.waitKey(1)
        if k == 27:  
            break
        if count >= total_samples:
            break

cap.release()
cv2.destroyAllWindows()
