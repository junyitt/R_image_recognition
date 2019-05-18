import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

def rotate_img(img, rotation_degree):
    rows,cols,channels = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_degree,1)
    rot_img = cv2.warpAffine(img,M,(cols,rows))
    return rot_img

def crop_first_face(img, faces):
    faces = face_cascade.detectMultiScale(img, 1.3, 5) #Face Detection
    largest_face_index = np.argmax(faces[:,3])
    x,y,w,h = faces[largest_face_index]
    crop_img = img[y:y+h, x:x+w]
    return crop_img
        
def get_eyes_shape(img, rotation_degree):
    try:
        rot_img = rotate_img(img, rotation_degree) # Rotate image by n degrees
        gray_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY) # Transform image to grayscale
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5) # Face Detection
        if len(faces) == 0:
            return 0
        else: 
            crop_img = crop_first_face(rot_img, faces)
        
        eyes = eye_cascade.detectMultiScale(crop_img)
        
        return eyes.shape[0]
    except Exception as err:
#         print(err)
        return 0

def get_correct_rdeg(img, threshold = 2):
    counter = 0 
    previous = 0
    for rotation_degree in np.arange(-30, 150, 5):
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
        crop_img = crop_first_face(img, faces)
        output_img = crop_img
    except Exception as err:
#         print(err)
        output_img = None
    return output_img


def get_crop_img(img_path, return_none = True):
    crop_img = get_crop_img_without_rotation(img_path)
    if crop_img is None:
        try:
            img = cv2.imread(img_path)
            cr = get_correct_rdeg(img)
            rot_img = rotate_img(img, cr)
            gray = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY) #Transform image to grayscale
            faces = face_cascade.detectMultiScale(gray, 1.3, 5) #Face Detection
            print(faces)
            crop_img = crop_first_face(img, faces)
            return crop_img
        except Exception as err:
#             print(err)
            if return_none:
                return None
    else:
        return crop_img

    

