#coding=utf-8

import cv2

import time
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import json

import sys
import imp


imp.reload(sys)

#寻找车牌左右边界

def find_edge(image):
    sum_i = image.sum(axis=0)
    sum_i =  sum_i.astype(np.float)
    sum_i/=image.shape[0]*255
    # print sum_i

    start= 0 ;
    end = image.shape[1]-1

    for i,one in enumerate(sum_i):
        if one>0.4:
            start = i;
            if start-3<0:
                start = 0
            else:
                start -=3

            break;
    for i,one in enumerate(sum_i[::-1]):

        if one>0.4:
            end = end - i;
            if end+4>image.shape[1]-1:
                end = image.shape[1]-1
            else:
                end+=4
            break
    return start,end


#垂直边缘检测
def verticalEdgeDetection(image):
    image_sobel = cv2.Sobel(image.copy(),cv2.CV_8U,1,0)
    # image = auto_canny(image_sobel)

    # img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT
    # canny_image  = auto_canny(image)
    flag,thres = cv2.threshold(image_sobel,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    print(flag)
    flag,thres = cv2.threshold(image_sobel,int(flag*0.7),255,cv2.THRESH_BINARY)
    # thres = simpleThres(image_sobel)
    kernal = np.ones(shape=(3,15))
    thres = cv2.morphologyEx(thres,cv2.MORPH_CLOSE,kernal)
    return thres


#确定粗略的左右边界
def horizontalSegmentation(image):

    thres = verticalEdgeDetection(image)
    # thres = thres*image
    head,tail = find_edge(thres)
    # print head,tail
    # cv2.imshow("edge",thres)
    tail = tail+5
    if tail>135:
        tail = 135
    image = image[0:35,head:tail]
    image = cv2.resize(image, (int(136), int(36)))
    return image


#打上boundingbox和标签
def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1, cv2.LINE_AA)

    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    #draw.text((int(rect[0]+1), int(rect[1]-16)), addText.decode("utf-8"), (255, 255, 255), font=fontC)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)

    return imagex





