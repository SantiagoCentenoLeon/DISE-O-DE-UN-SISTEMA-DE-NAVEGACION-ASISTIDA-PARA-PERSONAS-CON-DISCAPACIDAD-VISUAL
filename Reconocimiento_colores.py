#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:37:27 2019

@author: edmar
"""

import cv2
import numpy as np
import time
cap = cv2.VideoCapture(2)

w=640
h=480
e=200

if not cap.isOpened():
    print("Cannot open camera")
    exit()

lower_blue = np.array([50,40,40])
upper_blue = np.array([70,255,255])

while True:
    
    # Take each frame
    _, frame = cap.read()
    #cv2.imwrite("foto.jpg",frame)
    #frame=cv2.imread("foto.jpg")

    #time.sleep(1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    average=cv2.blur(hsv,(10,10))
    
    # define range of blue color in HSV
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(average, lower_blue, upper_blue)

    #Dilate
    mask = cv2.erode (mask,cv2.getStructuringElement(cv2.MORPH_RECT,(6,6)),iterations = 1)
    #mask = cv2.dilate (mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations = 1)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #Deteeccion contornos
    contornos, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contornos, -1, (255,0,0), 3)
    flag_riego=False
    for c in contornos:
            area = cv2.contourArea(c)
            if area > 50:
                cv2.drawContours(frame, c, -1, (255,0,0), 3)
                M = cv2.moments(c)
                if (M["m00"]==0): M["m00"]=1
                x = int(M["m10"]/M["m00"])
                y = int(M['m01']/M['m00'])
                if x in range(int((w-e)/2),int((w+e)/2)) and y in range(int((h-e)/2),int((h+e)/2)):
                    flag_riego=True

    print(flag_riego)

    cv2.imshow('frame',frame)
    cv2.imshow('Mascara',mask)
    cv2.imshow('Filtrado',res)
    #cv2.imshow('mask',mask)
    #cv2.imshow('mask',mask)
    
    #cv2.imshow('res',res)
    if cv2.waitKey(30) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
