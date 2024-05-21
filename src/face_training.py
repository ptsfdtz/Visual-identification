import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm

def getImagesAndLabels(path, folder_name, start_id):
    folder_path = os.path.join(path, folder_name)
    imagePaths = []
    for dirpath, _, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                imagePaths.append(os.path.join(dirpath, filename))
    
    faceSamples = []
    ids = []

    for imagePath in tqdm(imagePaths, desc=f"Processing images in folder '{folder_name}'"):  
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        faces = detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(start_id)  # 使用分配的ID
    return faceSamples, ids

path = 'Facedata'
cascade_path = 'data/haarcascade_frontalface_default.xml'
trainer_file = 'face_trainer/trainer.yml'

recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.isfile(trainer_file):
    recognizer.read(trainer_file)

if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

detector = cv2.CascadeClassifier(cascade_path)

folders_to_train = []
current_max_id = -1

if os.path.isfile(trainer_file):
    # 获取当前最大ID
    labels = recognizer.getLabelsByString()
    if labels:
        current_max_id = max(labels.keys())

# 获取未训练的文件夹
for folder_name in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder_name)):
        trainer_file_folder = f'face_trainer/{folder_name}_trainer.yml'
        if not os.path.isfile(trainer_file_folder):
            folders_to_train.append(folder_name)

all_faces = []
all_ids = []

for folder_name in folders_to_train:
    current_max_id += 1
    print(f'Training faces in folder "{folder_name}". It will take a few seconds. Wait ...')
    faces, ids = getImagesAndLabels(path, folder_name, current_max_id)
    all_faces.extend(faces)
    all_ids.extend(ids)
    print(f"{len(np.unique(ids))} faces trained for folder '{folder_name}'.")

# 训练所有新数据
if all_faces:
    recognizer.update(all_faces, np.array(all_ids))

    if not os.path.exists('face_trainer'):
        os.makedirs('face_trainer')
    recognizer.write(trainer_file)
    print(f"{len(np.unique(all_ids))} faces trained. Exiting Program.")
else:
    print("All folders have been trained. Exiting Program.")
