from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import os
from deepface import DeepFace   

def detect_faces(data):
    file = "frame813.jpg"
    img_path = os.path.join('data', file)
    img = cv2.imread(img_path)
    obj = RetinaFace.detect_faces(img_path)
    len(obj.keys())
    for key in obj.keys():
        identity = obj[key]
        facial_area = identity["facial_area"]
        cv2.rectangle(img, (facial_area[2], facial_area[3]),(facial_area[0], facial_area[1]), (255, 255, 255), 1)

    plt.imshow(img)
    plt.show()

def recog_faces(input):
    obj = DeepFace.verify(img1_path = "./data/frame813.jpg", img2_path = input, model_name = "ArcFace", detector_backend = "retinaface")
    print(obj)

detect_faces("data")
input = './Justin.png'
recog_faces(input)
