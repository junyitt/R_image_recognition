library(reticulate)

py_run_string(
'
import numpy as np
import cv2
import os

def get_crop_img(img_path):
    try:
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Transform image to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face Detection
        x,y,w,h = faces[0]
        crop_img = img[y:y+h, x:x+w]
        output_img = crop_img
    except:
        output_img = None
    return output_img
'
)

py_run_file("crop.py")

# # Example:
cv2 <- import("cv2")
output_img <- py$get_crop_img("Data/_insta/debbie6.jpg")
cv2$imwrite('output.jpg',output_img)

