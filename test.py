## importing libraries

import cv2
import sys
from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifie
from keras.models import load_model
import tensorflow as tf
import numpy as np
import math
import os
import time
import tensorflow as tf
import csv

## accessing the camera

## to give default value for camera device index
s=0

## to check if there is command line specification to overide that default value
if len(sys.argv) > 1 :
    s = sys.argv[1]

## create a class to capture video from the give ncamera index
cap = cv2.VideoCapture(s)

## creating a window which will send the output to
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_FULLSCREEN)

## creating a detector class which will define parameters for hand detection
detector = HandDetector(maxHands=1)

## load the model

## creating offset to get padding in crop window
offset = 20

## image size for the image white background matrix
imgSize = 224

## for the display text
dispText="Enter the specific key to store symbol"

## counter for no. of images captured
counter = 0

## file path for data folder
file_path = "C:\\E Drive\\f drive\\rishabh ki folder hain delete mat karna\\utu\\sem-7\\Project\\Data\\"

## load labels
label_save_path = "./Model/label.csv"
with open(label_save_path, 'r') as f:
    rd = csv.reader(f)
    labels = list(rd)

# Load the model with custom object scope
model = load_model("Model/model.h5", compile=False)

## create a while loop which will alow us to continuously stream the video
while cv2.waitKey(1) != 27: ## runs unless user hits escape key
    success, img = cap.read() ## read method will return a single frame from the video stream
    imgOutput = img.copy()
    # cv2.putText(img, dispText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    hands, img = detector.findHands(img) ## it will store the detected hand information for the given frame
    if hands:
        hand = hands[0] ## this crop multiple hands to single hands
        x, y, w, h = hand['bbox'] ## it will provide bounding box for hand

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 ## using this white image matrix to resize the long and uneven width and height cropped image  
        imgCrop = img[y-offset:y + h+offset,x-offset:x + w+offset] ## it will crop the hand from the whole frame

        imgCropShape = imgCrop.shape

        aspectRatio = h/w ## here we are calculating the aspect ratio of cropped image

        ## In this we are adjusting the cropped image to a fixed image of 300x300 pixels
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - min(hCal, imgSize))/2)
            imgWhite[hGap:min(hCal + hGap, imgSize), :] = imgResize[:min(hCal, imgSize), :]
        
        # Resize the imgWhite to 224x224
        imgWhiteResized = cv2.resize(imgWhite, (224, 224))

        # Normalize the image (optional but recommended)
        imgWhiteResized = imgWhiteResized / 255.0

        # Add batch dimension
        imgWhiteResized = np.expand_dims(imgWhiteResized, axis=0)

        # Predicts the model
        prediction = model.predict(imgWhiteResized)
        index = np.argmax(prediction)
        class_name = labels[index]
        confidence_score = prediction[0][index]

        cv2.putText(imgOutput, class_name[0],(x, y -20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0 ,255), 2)

        # Print prediction and confidence score
        # print("Class:", class_name[0], end=" ")
        # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)

    if not success:
        break
    cv2.imshow(win_name, imgOutput)




## it will end the video window
cap.release()
cv2.destroyWindow(win_name)