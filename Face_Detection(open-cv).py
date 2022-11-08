#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
alg="haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(alg)
vs=cv2.VideoCapture(0)


# In[ ]:


while True:
    _,img=vs.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.imshow('face',img)
        key=cv2.waitKey(0)
        if key==27:
            break


# In[ ]:




