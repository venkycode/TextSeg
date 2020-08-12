import cv2
import sys
import numpy as np
import os
import heapq
import random
import math

folder="test data"
input_img = "im20.png"
img_path=folder+"/"+input_img
img = cv2.imread(img_path)
height = img.shape[0]
width = img.shape[1]
header_position1 = -1
header_position2 = -1
part_projections = []
offset = 20
parts = 40
part_height=-1


def getVerticalProjectionProfile(image, start_position=0, end_position=height):
    x = [[0 for i in range(width)] for j in range(height)]

    for i in range(start_position, end_position):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    vertical_projection = np.sum(x, axis=0)

    return vertical_projection


def getHorizontalProjectionProfile(image):

    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]

    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    horizontal_projection = np.sum(x, axis=1)

    return horizontal_projection


def cropROI(image):
    horizontal_projection = getHorizontalProjectionProfile(image)
    verticle_projection = getVerticalProjectionProfile(image)
    start_x = -1
    end_x = width
    start_y = -1
    end_y = height
    for i in range(width):
        if verticle_projection[i]:
            start_x = i-1
            break
    for i in range(width-1, 0, -1):
        if verticle_projection[i]:
            end_x = i+1
            break
    for i in range(height):
        if horizontal_projection[i]:
            start_y = i-1
            break
    for i in range(height-1, 0, -1):
        if horizontal_projection[i]:
            end_y = i+1
            break
    image = image[start_y:end_y, start_x:end_x]
    return image


def removeHeader(img):
    global header_position1
    global header_position2
    pro = getHorizontalProjectionProfile(img)
    mx = max(pro)
    for i in range(height):
        if(abs(mx-pro[i]) <= mx*0.30):
            if header_position1 == -1:
                header_position1 = i
            header_position2 = i
            for j in range(width):
                img[i][j] = 255
    for i in range(max(0, header_position1-5), header_position1):
        for j in range(width):
            img[i][j] = 255
    for i in range(header_position2,header_position2+5):
        for j in range(width):
            img[i][j]=255
    # correctHeader(img,header_position1,header_position2)
    #cv2.imshow('output.png', img)
    # cv2.waitKey(0)
    return img


def getSources():
    global part_projections
    sources = []
    l = -1
    for j in range(1, width):
        header_part = math.ceil((header_position2+1)/part_height)
        #print(header_part)
        if part_projections[header_part][j-1] and part_projections[header_part][j] == 0:
            l = j
        if part_projections[header_part][j] and part_projections[header_part][j-1] == 0 and l != -1:
            sources.append(l+(j-l)//2)
    return sources


def getPath(sources):
    segments = []
    for source in sources:
        header_part = math.ceil((header_position1+1)/part_height)
        tmp_segment = [source]
        success = 0
        for i in range(header_part, parts):
            success = 0
            for k in range(offset):
                if tmp_segment[-1]+k < width and part_projections[i][tmp_segment[-1]+k] == 0:
                    tmp_segment.append(tmp_segment[-1]+k)
                    success = 1
                    break
                if tmp_segment[-1] >= k and part_projections[i][tmp_segment[-1]-k] == 0:
                    tmp_segment.append(tmp_segment[-1]-k)
                    success = 1
                    break
            if success == 0:
                segments.append([])
                break
        if success:
            tmp_segment.reverse()
            for i in range(header_part-1, 0, -1):
                success = 0
                for k in range(offset):
                    if tmp_segment[-1]+k < width and part_projections[i][tmp_segment[-1]+k] == 0:
                        tmp_segment.append(tmp_segment[-1]+k)
                        success = 1
                        break
                    if tmp_segment[-1] >= k and part_projections[i][tmp_segment[-1]-k] == 0:
                        tmp_segment.append(tmp_segment[-1]-k)
                        success = 1
                        break
                if success == 0:
                    segments.append([])
                    break
            if success:
                tmp_segment.reverse()
                segments.append(tmp_segment)
    return segments


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# print(img)
img = cropROI(img)
# cv2.imshow("tmp.jpg",img)
# cv2.waitKey(0)
height = img.shape[0]
width = img.shape[1]

img = removeHeader(img)

print(height)
part_height = math.ceil(height/parts)
print(part_height)
for i in range(parts):
    tmp_projection = getVerticalProjectionProfile(
        img, part_height*i, min(part_height*(i+1), height))
    part_projections.append(tmp_projection)

#print(part_projections)

sources = getSources()
print(sources)
segments = getPath(sources)

print(segments)

tmpimg = img.copy()
for segment in segments:
    if len(segment)==0:continue
    for k in range(len(segment)):
        for i in range(part_height*k, min(part_height*(k+1), height)):
            tmpimg[i][segment[k]] = 100

cv2.imshow("tmpfinal.jpg", tmpimg)
cv2.waitKey(0)
cv2.imwrite("results2/final_parts_seg_"+input_img, tmpimg)