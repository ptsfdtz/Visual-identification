import numpy as np
from PIL import Image
import os
import cv2
from tqdm import tqdm

def getImagesAndLabels(path, folder_name):
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
        
        id_str = os.path.split(imagePath)[-1].split("_")[1]
        id_num = int(id_str)  
        
        faces = detector.detectMultiScale(img_numpy)
        
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x: x + w])
            ids.append(id_num)  
    
    return faceSamples, ids

path = 'Facedata'

recognizer = cv2.face.LBPHFaceRecognizer_create()

cascade_path = 'data/haarcascade_frontalface_default.xml'
if not os.path.isfile(cascade_path):
    raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

detector = cv2.CascadeClassifier(cascade_path)

folders_to_train = []
for folder_name in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder_name)):
        trainer_file = f'face_trainer/{folder_name}_trainer.yml'
        if not os.path.isfile(trainer_file):
            folders_to_train.append(folder_name)

for folder_name in folders_to_train:
    print(f'Training faces in folder "{folder_name}". It will take a few seconds. Wait ...')
    faces, ids = getImagesAndLabels(path, folder_name)

    recognizer.train(faces, np.array(ids))

    if not os.path.exists('face_trainer'):
        os.makedirs('face_trainer')
    recognizer.write(f'face_trainer/{folder_name}_trainer.yml')
    print(f"{len(np.unique(ids))} faces trained for folder '{folder_name}'.")

if not folders_to_train:
    print("All folders have been trained. Exiting Program.")
