import cv2
import sys
import numpy as np
import imutils
import os
import heapq
import random
from math import ceil
import glob

path='C:/Users/Venkatesh Kotwade/Desktop/Project/TextSeg/words'

def getHorizontalProjectionProfile(image):
    height= image.shape[0]
    width= image.shape[1]
    #print(height,width)
    x = [[0 for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    horizontal_projection = np.sum(x, axis=1)
    #print(horizontal_projection)
    return horizontal_projection

def getVerticalProjectionProfile(image):
    height= image.shape[0]
    width= image.shape[1]
    #print(height,width)
    x = [[0 for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    vertical_projection = np.sum(x, axis=0)
    #print(vertical_projection)
    return vertical_projection

def break_line(line):
    vertical_projection=getVerticalProjectionProfile(line)
    slice_list=list()
    sz=len(vertical_projection)
    for i in range(sz-1):
        if vertical_projection[i]==0 and vertical_projection[i+1]:
            slice_list.append(i)
        if vertical_projection[i] and vertical_projection[i+1]==0:
            slice_list.append(i+1)
    list_of_words=list()
    skip=0
    sz=len(slice_list)
    for i in range(sz-1):
        if not skip:
            new_img= line[:,slice_list[i]:slice_list[i+1]+1] 
            list_of_words.append(new_img)
            skip=1
        else:
            skip=0
    return list_of_words   


def process(filename):
    img= cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
    cv2.imwrite("fin.jpg",img)
    cv2.imshow("tmp",img)
    cv2.waitKey(0)
    horizontal_projection=getHorizontalProjectionProfile(img)
    #print(vertical_projection)
    sz=len(horizontal_projection)
    slice_list=list()
    for i in range(0,sz-1):
        #print(horizontal_projection[i])
        if horizontal_projection[i]==0 and horizontal_projection[i+1]:
            slice_list.append(i)
        if horizontal_projection[i] and horizontal_projection[i+1]==0:
            slice_list.append(i+1)
    

    print(img.shape)
    print(len(horizontal_projection))
    #for i in range(img.shape[0]):
     #   if not horizontal_projection[i]:
      #      for j in range(img.shape[1]):
       #         img[i][j]=0
    
    cv2.imwrite("fin1.jpg",img)
    print(slice_list)
    list_of_lines=list()
    skip=0
    sz=len(slice_list)
    for i in range(0,sz-1):
        if not skip:
            new_img= img[slice_list[i]:slice_list[i+1]+1,:]
            list_of_lines.append(new_img)
            skip=1
        else:
            skip=0
    cnt=0
    for i in list_of_lines:
        cv2.imwrite("file"+str(cnt)+".jpg",i)
        #cv2.imshow("tmp",i)
        #cv2.waitKey(0)
        cnt+=1 
    cnt=0
    for line in list_of_lines:
        if line.shape[0]<=10:
            continue
        list_of_words= break_line(line)
        for word in list_of_words:
            cur_h=word.shape[0]
            ratio= 200/cur_h
            dim=(int(word.shape[1]*ratio),int(word.shape[0]*ratio))
            upscaled_word=cv2.resize(word,dim,interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('./words/word'+str(cnt)+'.jpg',upscaled_word)
            print(cur_h,upscaled_word.shape[0],ratio,dim)
            cnt+=1
        

filename= glob.glob('./Test/Bangla (1)/Bangla/0257_AamarUrmi_Img_300_Org_Page_0007.tif')
print(filename)
process(filename[0])       