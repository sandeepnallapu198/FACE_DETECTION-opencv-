#!/usr/bin/env python
# coding: utf-8

# In[29]:


import face_recognition as FC
import numpy as np
import cv2

img=FC.load_image_file("C:\\AI(ml&dl)\\DATASETS\\FACE_DEC_DATASET\\Bill gates.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
faceLoc=FC.face_locations(img)[0]
imgEncod=FC.face_encodings(img)[0]
cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[1]),(255,0,255),2)

Testimg=FC.load_image_file("C:\\AI(ml&dl)\\DATASETS\\FACE_DEC_DATASET\\Bill gates.jpg")
Testimg=cv2.cvtColor(Testimg,cv2.COLOR_BGR2RGB)
TestFaceLoc=FC.face_locations(Testimg)[0]
TestimgEncod=FC.face_encodings(Testimg)[0]
cv2.rectangle(img,(TestFaceLoc[3],TestFaceLoc[0]),(TestFaceLoc[1],TestFaceLoc[1]),(255,0,255),2)

result=FC.compare_faces([imgEncod],TestimgEncod)
faceDis=FC.face_distance([imgEncod],TestimgEncod)
print(result,faceDis)
cv2.putText(Testimg,f'{result}{np.round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('img',img)
cv2.imshow('test_img',Testimg)
cv2.waitKey(0)


# In[ ]:




