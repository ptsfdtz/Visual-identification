import numpy as np
import cv2

# 人脸识别分类器
faceCascade = cv2.CascadeClassifier(r'data\haarcascade_frontalface_default.xml')

# 识别眼睛的分类器
eyeCascade = cv2.CascadeClassifier(r'data\haarcascade_eye.xml')

# 开启摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取摄像头中的图像，ok为是否读取成功的判断参数
    ok, img = cap.read()
    if not ok:
        print("Failed to read from camera. Exiting...")
        break
    
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(32, 32)
    )

    # 在检测人脸的基础上检测眼睛，并且只在检测到眼睛的情况下绘制人脸框
    for (x, y, w, h) in faces:
        fac_gray = gray[y: (y+h), x: (x+w)]
        
        eyes = eyeCascade.detectMultiScale(fac_gray, 1.3, 2)
        if len(eyes) > 0:
            # 如果检测到眼睛，则绘制人脸矩形框
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示图像
    cv2.imshow('video', img)

    # 按下ESC退出循环
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
