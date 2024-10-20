import os
import cv2
import numpy
import matplotlib.pyplot as plt
import pickle
# from tensorflow.keras.preprocessing import image

dataDir= "./Data"

data=[]
labels=[]

for dir_ in os.listdir(dataDir):
    for img_path in os.listdir(os.path.join(dataDir, dir_)):
        img = cv2.imread(os.path.join(dataDir,dir_,img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img_rgb)
        labels.append(dir_)

f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()