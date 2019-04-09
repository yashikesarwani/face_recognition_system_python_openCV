# Author-YASHI KESARWANI #


#Description- Haarcascade classifier is a library classifier which is used to detect object using several machine learning algorithm


#import modules
import cv2
import numpy as np

#define face classifier using haarcascade library classifier (It is inbuilt, you can download it or just install openCV)
face_classifier  = cv2.CascadeClassifier('C:/Users/hp/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


#function for taking input using webcam for getting the face of the user
def face_extractor(img):
    gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return None
    
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h,x:x+w]


    return cropped_face

#for capturing video which will be taken as collection of images
cap = cv2.VideoCapture(0)
count = 0

#when uh get face, we extract it, convert it to gray scale, resize it and store the cropped face.
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)


        #we send the extracted images to the selected path to store the input.
        file_name_path = 'C:/Users/hp/Downloads/faces/user' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper', face)
    else:
        print("face not found")
        pass

    # 13 is the ASCII value, if we press enter or the count value increases to 100 it breaks up
    if cv2.waitKey(1) == 13 or count == 100:
        break

#this is for the release of the camera.    
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!!!")
