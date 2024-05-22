import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2



image = cv2.imread('imageDataSet/images/virat_kohli/12b898ec07.jpg')
plt.imshow(image,cmap='grey')
plt.show()
face_cascade = cv2.CascadeClassifier()
face_cascade.load('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)







    
    
