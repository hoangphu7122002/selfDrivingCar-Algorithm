from gc import collect
from itertools import count
from turtle import left, right
from unity_utils.unity_utils import Unity
import cv2
import time
import imutils
from cmath import atan
import math
from collections import Counter
import collections
import numpy as np
from math import acos, degrees

#======================================
unity_api = Unity(11000)
unity_api.connect()
errors = 0
ti = time.time()
images = []

Left = np.array([
    [(150, 0), (599, 160), (599, 0)]
])
Right = np.array([
    [(450, 0), (0, 150), (0, 0)]
])

#======================================
def convertBinary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret,bin_image = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    
    kernel = np.zeros((10,10),np.uint8)
    # bin_image = cv2.dilate(bin_image, kernel, iterations=2)

    return bin_image

def closing(img):
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img,kernel,iterations = 5)
    img = cv2.erode(img,kernel,iterations = 2)
    return img

def draw_point(c,image):
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][-1])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    extMid=[]
    extMid.append(int((extRight[0] + extBot[0])/2))
    extMid.append(int((extRight[1] + extBot[1])/2))
    extMid = tuple(extMid)

    a,b = extMid
    extMid = (a,b)

    cv2.circle(image, extMid, 10, (200, 200, 200), -1)

    return image,extLeft,extRight,extTop,extBot,extMid

def find_counters(image):
    # image = convertBinary(image)
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    try:
        c = max(cnts,key=cv2.contourArea)
        c_area = cv2.contourArea(c)
    except ValueError:
        return image,(0,0),(0,0),(0,0),(0,0),c_area
    # determine the most extreme points along the contour
    
    image,l,r,t,b,m = draw_point(c,image)
    return image,l,r,t,b,m,c_area

#==========================MAIN FOCUS========================
def Direction(image):
    height = image.shape[0] - 50
    find_bound = [x for x,y in enumerate(image[height, :]) if y == 255] #find light
    center = (min(find_bound) + max(find_bound)) // 2 #find center of region
    error = image.shape[1] // 2 - center
    left_point = (min(find_bound),height)
    right_point = (max(find_bound),height)
    cv2.circle(image, left_point, 10, (200, 200, 200), -1)
    cv2.circle(image, right_point, 10, (200, 200, 200), -1)
    cv2.line(image,left_point,right_point,(200,200,200),2)    

    return error,(center,height)

def control(image,area):
    if area >= 13500:
        #narrow 
        cv2.fillPoly(image, Left, 0)
        cv2.fillPoly(image, Right, 0)
        # print('image: ',image)
        # image = images[-1]
    error,point_center = Direction(image)
    angle = computePD(error)
    speed = angle * 0.2 + 15
    return -angle, speed, point_center

def computePD(error, p=0.12, d=0.01):
    global ti
    global errors 

    delta_t = time.time() - ti
    ti = time.time()

    P = error * p
    D = (error - errors) / delta_t * d

    angle = P + D
    
    if abs(angle) > 6:
        #if left or right
        angle = np.sign(angle) * 40
    errors = error
    #if ahead, go ahead
    return int(angle) * 1/2
#==========================MAIN FOCUS========================

while True:
    start_time = time.time()
    left_image, right_image = unity_api.get_images()

    left_image = convertBinary(left_image)
    right_image = convertBinary(right_image)

    frame = np.concatenate([left_image,right_image],axis = 1)
    frame = frame.astype(np.uint8)
    print(frame.shape)
    try:
        frame,left_f,right_f,top_f, bot_f,mid_f,area = find_counters(frame)
    except:
        frame,left_f,right_f,top_f, bot_f,mid_f,area = frame,(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)

    unity_api.show_images(left_image,right_image)
    #cv2.imshow('predict',frame)
    angle_,speed, point_center = control(frame,area)
    cv2.circle(frame, point_center, 10, (200, 200, 200), -1)
    cv2.line(frame,point_center,mid_f,(200,200,200),2)
    data = unity_api.set_speed_angle(speed, angle_)
    print(data)
    print(area)
    cv2.imshow("combine",frame)