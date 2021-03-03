import cv2
import numpy as np
import dlib


cap=cv2.VideoCapture(0)
cap.set(10,150)
detector=dlib.get_frontal_face_detector()
while True:
    success,img=cap.read()
    vid=cv2.flip(img,1)
    vidgrey=cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    faces=detector(vidgrey)
    pred=dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
    for face in faces:
        x1= face.left()
        y1= face.top()
        x2= face.right()
        y2= face.bottom()
        landmarks=pred(vidgrey,face)
        for n in range(0,81):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            cv2.circle(vid,(x,y),2,(0,255,0),-1)

      #  cv2.rectangle(vid,(x1,y1),(x2,y2),(0,255,0),3)



    cv2.imshow("video",vid)
    if cv2.waitKey(1)&0xff==ord('q'):
        break