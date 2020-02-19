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


def cropROI(image):
    height=image.shape[0]
    width=image.shape[1]
    horizontal_projection=getHorizontalProjectionProfile(image)
    verticle_projection=getVerticalProjectionProfile(image)
    start_x=-1
    end_x=width
    start_y=-1
    end_y=height
    for i in range(width):
        if verticle_projection[i]:
            start_x= i-1
            break
    for i in range(width-1,0,-1):
        if verticle_projection[i]:
            end_x= i+1
            break
    for i in range(height):
        if horizontal_projection[i]:
            start_y= i-1
            break
    for i in range(height-1,0,-1):
        if horizontal_projection[i]:
            end_y= i+1
            break
    image= image[start_y:end_y, start_x:end_x]
    return image


def processSources(listSources,dif_factor):
    
    new_list=[]
    new_list.append(listSources[0][0])
    sz= len(listSources)
    min_tuple=(-1,99999999999999999)
    for i in range(sz):
        if listSources[i][1]<min_tuple[1]:min_tuple=listSources[i]
        if i!=sz-1 and listSources[i+1][0]-listSources[i][0]>=dif_factor:
            new_list.append(min_tuple[0])
            min_tuple=(-1,99999999999999999)
    new_list.append(listSources[sz-1][0])
            
    return new_list


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


img=cv2.imread("rotated_header_removed"+input_img);
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(img)
img=cropROI(img)
#cv2.imshow("tmp.jpg",img)
#cv2.waitKey(0)
cv2.imwrite("cropped_"+input_img,img)


vertical_projection= getVerticalProjectionProfile(img)
 
min_density= min(vertical_projection)
max_density= max(vertical_projection)
 
cutoff_factor= 0.2 #adjustable
 
cutoff_density= (max_density-min_density)*cutoff_factor+ min_density
 
source_segment_points= list()
 
width=img.shape[1]
height= img.shape[0]
 
for i in range(width):
    if vertical_projection[i]<=cutoff_density:
        source_segment_points.append((i,vertical_projection[i]))
 
'''source_segment_points= processSources(source_segment_points,3)
source_segment_points= processSources(source_segment_points,10)
source_segment_points= processSources(source_segment_points,20)
print(source_segment_points)'''
source_segment_points= processSources(source_segment_points,20)
#source_segment_points= processSources(source_segment_points,40)
#source_segment_points= processSources(source_segment_points,60)
 
# checker code
 
tmpimg=img.copy()
for i in source_segment_points:
    for j in range(height):
        tmpimg[j][i]= 0
 
#cv2.imshow("tmp.jpg",tmpimg)
cv2.imwrite("basic_seg"+input_img,tmpimg)
#cv2.waitKey(0)
#cv2.imshow("tmp1.jpg",img)
#cv2.waitKey(0)
 
#-------------------#