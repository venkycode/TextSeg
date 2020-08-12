import cv2
import sys
import numpy as np
from math import ceil
import math
 
input_img = './input.png'
img = cv2.imread(input_img)
img2 = img.copy()
 
 
def getVerticalProjectionProfile(image, header_position=0):
    height = image.shape[0]
    width = image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]
 
    for i in range(header_position+1, height):
        for j in range(width):
            if image[i][j] == 255:
                x[i][j] = 1
 
    vertical_projection = np.sum(x, axis=0)
 
    return vertical_projection
 
 
def getHorizontalProjectionProfile(image):
    height = image.shape[0]
    width = image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]
 
    for i in range(height):
        for j in range(width):
            if image[i][j] == 255:
                x[i][j] = 1
 
    horizontal_projection = np.sum(x, axis=1)
 
    return horizontal_projection
 
 
def cropROI(image):
    height = image.shape[0]
    width = image.shape[1]
    horizontal_projection = getHorizontalProjectionProfile(image)
    verticle_projection = getVerticalProjectionProfile(image)
    start_x = -1
    end_x = width
    start_y = -1
    end_y = height
    for i in range(width):
        if verticle_projection[i]:
 
            if(i):
                start_x = i-1
            else:
                start_x = i
            break
    for i in range(width-1, 0, -1):
        if verticle_projection[i]:
            if(width-1-i):
                end_x = i+1
            else:
                end_x = width-1
            break
    for i in range(height):
        if horizontal_projection[i]:
            if(i):
                start_y = i-1
            else:
                start_y = i
            break
    for i in range(height-1, 0, -1):
        if horizontal_projection[i]:
            if(height-1-i):
                end_y = i+1
            else:
                height = i
            break
    image = image[start_y:end_y, start_x:end_x]
    return image
 
 
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('im', img)
cv2.waitKey(0)
height = img.shape[0]
width = img.shape[1]
pro = getVerticalProjectionProfile(img)
 
 
print(img.shape)
psc = [0]
img1 = img.copy()
for i in range(1, width):
    if pro[i] < 2:
        psc.append(i)
        for j in range(height):
            img1[j][i] = 124

sm = 0
cnt = 0
print(psc)
sc = [0]
K = 10
for i in range(1, len(psc)):
    if psc[i]-psc[i-1] < K:
        sm += psc[i]
        cnt += 1
    elif cnt > 1:
        k = sm//cnt
        d1 = 0
        d2 = 0
        e = k
        while pro[e] >= 2:
            d1 += 1
            e -= 1
        e = k
        while pro[e] >= 2:
            d2 += 1
            e += 1
        if d1 < d2:
            k = e-d1
        else:
            k = e+d2
        # if k-sc[-1]<K:continue
        sc.append(k)
        for j in range(height):
            img[j][k] = 124
        sm = psc[i]
        cnt = 1
if cnt:
    k = sm//cnt
    d1 = 0
    d2 = 0
    e = k
    while pro[e] >= 2:
        d1 += 1
        e -= 1
    e = k
    while pro[e] >= 2:
        d2 += 1
        e += 1
    if d1 < d2:
        k = e-d1
    else:
        k = e+d2
    sc.append(k)
    for j in range(height):
        img[j][k] = 124

print(sc)
cv2.imshow("s1", img1)
cv2.imwrite("res2.png", img)
 
cv2.waitKey(0)