#OpenCV module
import cv2
import matplotlib.pyplot as plt
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

subjects = ["", "Elvis Presley", "Bob Dylan","others"]
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5 ,minSize=(20,20));
#if no faces are detected then select Region of Image manually
 if (len(faces) == 0):

     print("No face detected")
     r = cv2.selectROI(gray)
     faces = r
     (x, y, w, h) = faces
     return gray[y:y+w, x:x+h], faces

#under the assumption that there will be only one face,
#extract the face area
 (x, y, w, h) = faces[0]
 # cv2.imshow('pic',faces[0])
 print(faces[0])
#return only the face part of the image
 return gray[y:y+w, x:x+h], faces[0]
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w,  y+h), (0, 255, 0), 2)

def draw_text(img, text,x, y):
    cv2.putText(img, text,(x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
def predict(test_img):

    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('trainer/trainer.yml')
    # predict the image using our face recognizer
    label, confidence = recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img
print("Predicting images...")

#load test images
test_img = cv2.imread("test-data/images.jpeg")
#perform a prediction
predicted_img = predict(test_img)
print("Prediction complete")

#display the image

cv2.imshow(subjects[2], cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()