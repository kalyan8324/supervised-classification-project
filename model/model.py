import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import shutil

image_path = '/home/maren/Downloads/Python-project/datasets/crickte-players/images/virat_kohli/5bc384d5d8.jpg'
face_cascade_path = '/home/maren/Downloads/Python-project/cricket-CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_frontalface_default.xml'
eye_cascade_path = '/home/maren/Downloads/Python-project/cricket-CelebrityFaceRecognition/model/opencv/haarcascades/haarcascade_eye.xml'
path_to_dir = '/home/maren/Downloads/Python-project/cricket-CelebrityFaceRecognition/model/imageDataSet/images/'
to_path_dir = '/home/maren/Downloads/Python-project/cricket-CelebrityFaceRecognition/model/imageDataSet/cropped_imges/'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty():
    print("Failed to load face cascade")
if eye_cascade.empty():
    print("Failed to load face cascade")
def get_cropped_images_if_2_eyes(img):
    if not os.path.exists(img):
        print("error while loading image")
        return None
    image  = cv2.imread(img)
    if image is None:
        print("error while loading image")
        return None
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , 1.3,5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color
        else:
            return None

new_img_path = '/home/maren/Downloads/Python-project/datasets/crickte-players/images/virat_kohli/7da6647368.jpg'
cropped_img = get_cropped_images_if_2_eyes(new_img_path)

img_dir = img_dir = [entry.name for entry in os.scandir(path_to_dir) if entry.is_dir()]
cropped_img_dir = []
celebrity_file_name_dic = {}
for imgs in img_dir:
    print(imgs)
    count = 1
    celebrity_file_name_dic[imgs] = []
    for img in os.listdir(path_to_dir + imgs):
        crop_imges = get_cropped_images_if_2_eyes(path_to_dir + imgs + '/' + img)
        if crop_imges is not None:
            cropped_folder = to_path_dir + imgs
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_dir.append(cropped_folder)
            cropped_file_name = imgs + str(count)+".png"
            cropped_file_path = cropped_folder + "/" +cropped_file_name
            cv2.imwrite(cropped_file_path, crop_imges)
            celebrity_file_name_dic[imgs].append(cropped_file_path)
            count += 1
# for entry in os.scandir(path_to_dir):
#     if entry.is_dir():
#         img_dir.append(entry.path)       
# if os.path.exists(to_path_dir):
#     shutil.rmtree(to_path_dir)
# os.mkdir(to_path_dir)
# print(img)
if cropped_img is not None:
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image with Two Eyes Detected')
    plt.axis('off')
    plt.show()
else:
    print("No face detected")
    
# if not os.path.exists(image_path):
#     print("no image available")
# else:
#     print(f"Loading image")
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     # plt.imshow(gray,cmap='gray')
#     if not os.path.exists(face_cascade_path):
#         print("no face cascade file available")
#     if not os.path.exists(eye_cascade_path):
#         print("no eye cascade file available")
#     face_cascade = cv2.CascadeClassifier(face_cascade_path)
#     print("face_cascade:",face_cascade)
#     eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
#     if face_cascade.empty():
#         print("Failed to load face cascade")
#     if eye_cascade.empty():
#         print("Failed to load face cascade")
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
#     print("faces:",faces)
#     if len(faces) == 0:
#         print("No faces detected!!..")
#     else:
#         print(f"Detected {len(faces)} face(s)")
#         for (x,y,w,h) in faces:
#             cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = image[y:y+h, x:x+w]
#             eyes = eye_cascade.detectMultiScale(roi_gray)
#             for (ex,ey,ew,eh) in eyes:
#                 cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) 
#         cv2.imshow('img',image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         plt.title('face recognition')
#         # plt.show()
       
   
#     new_image = '/home/maren/Downloads/Python-project/datasets/crickte-players/images/virat_kohli/7da6647368.jpg'
#     original_image = cv2.imread(new_image)
#     cropped_image = get_cropped_images_if_2_eyes(new_image)
#     if cropped_image is not None:
#         plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
#         plt.title('Cropped Image with Two Eyes Detected')
#         plt.axis('off')
#         plt.show()
#     else:
#         print("No face detected")
#     plt.show()


            


    









    
    
