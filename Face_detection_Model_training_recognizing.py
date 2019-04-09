# @Author-YASHI KESARWANI  #


#import modules and inbuilt functions
import cv2
import numpy as np

#lisdir is used to make a list (in python) of all the files present in a given path
from os import listdir

#isfile is used to check whether a file is present in a given path.
from os.path import isfile, join

#define the path where uh have stored your input as extracted images of the user.
data_path = 'C:/Users/yashi/Downloads/faces/'

#making an array of all the files present in data_path
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

#initializing two lists
Training_Data, Labels = [], []


#reading image files in datapath and training the data using openCV inbuilt functions 
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype = np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype = np.int32)
model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Completed")

face_classifier = cv2.CascadeClassifier('C:/Users/yashi/AppData/Local/Programs/Python/Python37-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


#function for recognition of face taken in input, we will again take the image as input
# and recognize it whether it is the same user or not.(roi- region of interest)
def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return img, []
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi
#again capturing the video using webcam
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        #calculating confidence value
        if result[1]<500:
            confidence = int (100*(1-(result[1])/300))
            display_string = str(confidence) + '%Confidence it is user'
        cv2.putText(image, display_string, (100, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (250,120,255), 2)


        #if conf >75 means it is the same user.. so phone gets unlocked as in my ONE+6T :)
        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', image)

        #if conf <75 it remains locked.
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Cropper', image)

    #if face is not found
    except:
        cv2.putText(image, "Face not found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass


    if cv2.waitKey(1) == 13:
        break


cap.release()
cv2.destroyAllWindows()
        
#### your face recognizer system is ready ########
