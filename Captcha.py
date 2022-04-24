import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array
import json
from tensorflow.keras.models import load_model

f = open("info.txt", "r")
info = f.read()
info = json.loads(info)

def test(img_path) :   
    model = load_model("Captcha_model") 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, 0)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,2), np.uint8))
    img = cv2.dilate(img, np.ones((2,2), np.uint8), iterations = 1)
    img = cv2.GaussianBlur(img, (1,1), 0)
    image_list = [img[10:50, 30:50], img[10:50, 50:70], img[10:50, 70:90], img[10:50, 90:110], img[10:50, 110:130]]
    test_X = []
    for i in range(5) :
        test_X.append(img_to_array(Image.fromarray(image_list[i])))
    
    test_X = np.array(test_X)
    test_X/= 255.0
    
    test_y = model.predict(test_X)
    test_y = np.argmax(test_y, axis = 1)
    
    for res in test_y :
        print(info[str(res)])
    print(img_path[-9:])
    
test('dataset/2bg48.png')