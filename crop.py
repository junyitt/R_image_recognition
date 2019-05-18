import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
def get_eyes_shape(img, rotation_degree):
    try:
        rows,cols,channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_degree,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #Transform image to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face Detection
        if len(faces) == 0:
            return 0
        eyes = eye_cascade.detectMultiScale(gray)
        
        return eyes.shape[0]
    except: 
        return 0

def get_correct_rdeg(img, threshold = 2):
    counter = 0 
    previous = 0
    for rotation_degree in np.arange(-30, 330, 5):
        current = get_eyes_shape(img, rotation_degree)
        if current == 2:
            counter = counter + 1
        else:
            counter = 0
        if counter > threshold:
            return rotation_degree
    
    return None

def get_crop_img_without_rotation(img_path):
    try:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Transform image to grayscale
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face Detection
        x,y,w,h = faces[0]
        crop_img = img[y:y+h, x:x+w]
        output_img = crop_img
    except:
        output_img = None
    return output_img


def get_crop_img(img_path):
    crop_img = get_crop_img_without_rotation(img_path)
    if crop_img is None:
        try:
            img = cv2.imread(img_path)
            cr = get_correct_rdeg(img)
            print(cr)
            rows,cols,channels = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2), cr,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY) #Transform image to grayscale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face Detection
            x,y,w,h = faces[0]
            crop_img = dst[y:y+h, x:x+w]
            return crop_img
        except:
            return None
    else:
        return crop_img

    

