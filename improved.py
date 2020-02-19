import cv2
import sys
import numpy as np
import imutils
import os


input_img= "img2.jpg"
img=cv2.imread(input_img)
 
def getVerticalProjectionProfile(image,header_position=0): 
    height=image.shape[0]
    width=image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]
 
    for i in range(header_position+1,height):
        for j in range(width):
            if image[i][j]==0:x[i][j]=1
  
    vertical_projection = np.sum(x, axis = 0) 
  
 
    return vertical_projection 
 
def getHorizontalProjectionProfile(image): 
  
        # Convert black spots to ones 
    height=image.shape[0]
    width=image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]
 
    for i in range(height):
        for j in range(width):
            if image[i][j]==0:x[i][j]=1
  
    horizontal_projection = np.sum(x, axis = 1)  
  
    return horizontal_projection 


if os.path.isfile("rotated_header_removed"+input_img)==0:
    print("here")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    final_img=list(list())
    mx=0
    final_pro=list()
    for mult in range(-7,7):
        #rotated = imutils.rotate_bound(img, 0.5* mult)
        rows = img.shape[0]
        cols = img.shape[1]
 
        img_center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(img_center,mult*0.5, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255,255,255))
        pro=getHorizontalProjectionProfile(rotated)
        
        if mx<max(pro):
            final_img=rotated
            mx=max(pro)
            final_pro=pro
    print(mx)
    img=final_img
    height=img.shape[0]
    width=img.shape[1]
    pro=final_pro
    header_position1=-1
    header_position2=-1
    for i in range(height):
        if(abs(mx-pro[i])<=mx*0.30):
            if header_position1==-1:
                header_position1=i
            header_position2=i
            for j in range(width):img[i][j]=255
    #correctHeader(img,header_position1,header_position2)
    cv2.imshow('output.png',img)
    cv2.waitKey(0)         
    cv2.imwrite("rotated_header_removed"+input_img,img)