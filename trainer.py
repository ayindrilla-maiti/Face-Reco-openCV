#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np
#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Elvis Presley", "Bob Dylan","others"]
#function to detect face using OpenCV
def detect_face(img):
#convert the test image to gray scale as opencv face detector expects gray images
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#load OpenCV face detector, I am using Haar classifier which is slow but accurate
#there is also a more: LBP classifier
 face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

#let's detect multiscale images(some images may be closer to camera than others)
#result is a list of faces
 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

#if no faces are detected then detect manuslly using ROI
 if (len(faces) == 0):
     print("No face detected")
     r = cv2.selectROI(gray)
     faces = r
     (x, y, w, h) = faces
     return gray[y:y + w, x:x + h], faces

#
 (x, y, w, h) = faces[0]

#return only the face part of the image
 return gray[y:y+w, x:x+h], faces[0]


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def prepare_training_data(data_folder_path):
    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containin images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # ------STEP-3--------
        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)
            # print(face, rect)

            # ------STEP-4--------
            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # for f in faces:
                #     print(f)
                # add label for this face
                labels.append(label)
                # for f in labels:
                #  print(f)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
assure_path_exists('trainer/')
# face_recognizer.save('trainer/trainer.yml')
for t in range(100000):
 face_recognizer.train(faces, np.array(labels))
 face_recognizer.save('trainer/trainer.yml')
 print("Executing Training Step :",t)