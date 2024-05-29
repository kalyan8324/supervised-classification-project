#  Steps: load image into cv2 open_cv
#  data preprocessing 
#  1. convert to grayscale
#  2. blur image
#  3. threshold image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
# import shutil
import pywt 


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

# new_img_path = '/home/maren/Downloads/Python-project/datasets/crickte-players/images/virat_kohli/7da6647368.jpg'
# cropped_img = get_cropped_images_if_2_eyes(new_img_path)

img_dir = img_dir = [entry.name for entry in os.scandir(path_to_dir) if entry.is_dir()]
cropped_img_dir = []
celebrity_file_name_dic = {}
for imgs in img_dir:
    count = 1
    celebrity_file_name_dic[imgs] = []
    image_path_dc = os.path.join(path_to_dir,imgs)
    for img in os.listdir(image_path_dc):
        img_path = os.path.join(image_path_dc, img)
        crop_imges = get_cropped_images_if_2_eyes(path_to_dir + imgs + '/' + img)
        crop_imges = get_cropped_images_if_2_eyes(img_path)
        if crop_imges is not None:
            # cropped_folder = to_path_dir + imgs
            cropped_folder = os.path.join(to_path_dir, imgs)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_img_dir.append(cropped_folder)
            # cropped_file_name = imgs + str(count)+".png"
            cropped_file_name = f"{imgs}{count}.png"
            cropped_file_path = os.path.join(cropped_folder,cropped_file_name)
            cv2.imwrite(cropped_file_path, crop_imges)
            celebrity_file_name_dic[imgs].append(cropped_file_path)
            count += 1
        

def w2d(img,mode='haar',level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor(imArray,cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray = np.float32(imArray)
    imArray = imArray/255
    # compute coefficients
    coeffs=pywt.wavedec2(imArray,mode,level=level)
    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H,mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Manually examine cropped folder and delete any unwanted images

celebrity_file_names_dict = {}
for img_dir in cropped_img_dir:
    file_list = []
    for file in os.listdir(img_dir):
        file_list.append(file.path)
    celebrity_file_names_dict[img_dir] = file_list
print("celebrity_file_name_dic =>",celebrity_file_names_dict)

class_dict = {}
count = 0
for celebrity_name in celebrity_file_name_dic.keys():
    class_dict[celebrity_name] = count
    count = count + 1
print("class dic =>",class_dict)
# cropped_image used to model training 
x,y = [],[]
for celebrity_name, trainig_file in celebrity_file_name_dic.items():
    for file in trainig_file:
        img = cv2.imread(file)
        scalled_raw_img = cv2.resize(img,(32,32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(celebrity_name)
x = np.array(x).reshape(len(x), 4096).astype(float)
y = np.array(y)

#  TARINING OUR MODEL WE CAN USE SVM FOR PREDICTING OUR MODEL 

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#  spliting training set and test set data
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=0)
#  to scale our model 
#  here parameter randamly choosing
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf',C=10))])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# classification_report(y_test,pipe.predict(X_test))
# len(X_test)
# GridSearchCV to try out different models with different parameters .Goals is to come up with best model with best fine tuned parameter
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}
#  to find best model with best parameter
import pandas as pd
scores = []
best_estimators = {}
for mn , mp in model_params.items():
    # Create the pipeline with a StandardScaler and the model
    pipe = make_pipeline(StandardScaler(),mp['model'])
    #  to find best parameter
    gs = GridSearchCV(pipe,mp['params'],cv=5,return_train_score=False)
    gs.fit(X_train,y_train)
    scores.append({
        'model':mn,
        'best_score':gs.best_score_,
        'best_params':gs.best_params_
    })
    #  best estimator
    best_estimators[mn] = gs.best_estimator_
 #  To frame our model 
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(df)
sm = best_estimators['svm']
rf = best_estimators['random_forest']
lg = best_estimators['logistic_regression']

# print("svm",sm)
sm_cm = confusion_matrix(y_test,sm.predict(X_test))

# print("img_dir ::::",img_dir)
plt.figure(figsize = (10,7))
sns.heatmap(sm_cm, annot=True,fmt="d")
plt.title('SVM Confusion Matrix')
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

import joblib
# Save the model as a pickle in a file 
joblib.dump(sm,'svm.pkl')
# Save class dictionary
import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))




# for model_name, model_params in model_params.items():
#     #  first to scale our model
#     pipe = make_pipeline(StandardScaler(),model_params['model'])
#     # Create the GridSearchCV object with the pipeline
#     gd = GridSearchCV(pipe, model_params['params'], cv=5, return_train_score=False)
#     try:
#      gd.fit(X_train, y_train)
#     except ValueError as e:
#         print("Error for {model_name}: {e}")
#         raise Exception("invalid test and train details")
#     #  use the model to train best parameter
#     scores.append({
#         'model':model_name,
#         'best_score':gd.best_score_,
#         'best_params':gd.best_params_,
#     })
#     best_estimators[model_name] = gd.best_estimator_
#     print(model_name)
#     #  to print out the best model and best parameter
# df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
# print(df)
# print("best_estimators =>",best_estimators)
# #  to find the best model
# best_model = df['model'][df['best_score'].idxmax()]
# print(f"Best model is {best_model}")
# confusion_matrix(y_true, y_pred)
#  Building model using x , y varibles 
# for entry in os.scandir(path_to_dir):
#     if entry.is_dir():
#         img_dir.append(entry.path)       
# if os.path.exists(to_path_dir):
#     shutil.rmtree(to_path_dir)
# os.mkdir(to_path_dir)
# print(img)
# if cropped_img is not None:
#     plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
#     plt.title('Cropped Image with Two Eyes Detected')
#     plt.axis('off')
#     plt.show()
# else:
#     print("No face detected")
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


            


    









    
    
