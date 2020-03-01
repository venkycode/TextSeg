import cv2
import sys
import numpy as np
import os
import heapq
import random
from math import ceil
import glob

num_zone = 50
height = 0
width = 0
offset = 10
dp_mat = list()
isreachable = list()
projection_matrix = list()
header_offset = 0.2
header_position1=-1
header_position2=-1


def getHorizontalProjectionProfile(image):
    height = image.shape[0]
    width = image.shape[1]
    x = [[0 for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    horizontal_projection = np.sum(x, axis=1)
    return horizontal_projection


def remove_header(img):
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
    header_position2+=5
    header_position1-=5
    # correctHeader(img,header_position1,header_position2)
    #cv2.imshow('output.png', img)
    # cv2.waitKey(0)
    return img


def getVerticalProjectionProfile(image, start_position=0, end_position=height):
    x = [[0 for i in range(width)] for j in range(height)]

    for i in range(start_position, end_position):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    vertical_projection = np.sum(x, axis=0)

    return vertical_projection


def createProjectionMatrix(img):
    part_h = ceil(img.shape[0]/num_zone)
    part_projections = list()
    for i in range(num_zone):
        tmp_projection = getVerticalProjectionProfile(
            img, part_h*i, min(part_h*(i+1), height))
        part_projections.append(tmp_projection)
    return part_projections


def backtrack(i, j):
    part_h = ceil(height/num_zone)
    if i == 0:
        return
    rl = j  # right limit
    ll = j  # left limit
    for y in range(j, width):
        if projection_matrix[i][y]:
            break
        rl = y
    for y in range(j, 0, -1):
        if projection_matrix[i][y]:
            break
        ll = y

    ll = max(0, ll-(offset-1))
    rl = min(width-1, rl+(offset-1))
    for y in range(ll, rl+1):
        if projection_matrix[i-1][y] == 0:
            dp_mat[i-1][y] -= 1

    for y in range(ll, rl+1):
        if dp_mat[i-1][y] == 0 and projection_matrix[i-1][y] == 0 and isreachable[i-1][y] == 1:
            isreachable[i-1][y] = 0
            backtrack(i-1, y)
    return


def process(img):
    remove_header(img)
    global projection_matrix
    projection_matrix = createProjectionMatrix(img)
    global isreachable
    isreachable = [[0 for i in range(width)] for j in range(num_zone)]
    global dp_mat
    dp_mat = [[0 for i in range(width)] for j in range(num_zone)]
    for i in range(num_zone):
        for j in range(width):
            if projection_matrix[i][j]:
                isreachable[i][j] = 0
    for j in range(width):
        if projection_matrix[0][j] == 0:
            isreachable[0][j] = 1
    for i in range(num_zone-1):
        for j in range(width):
            if projection_matrix[i][j] or isreachable[i][j] == 0:
                continue
            for k in range(offset):
                if k+j >= width:
                    break
                if projection_matrix[i+1][j+k] == 0:
                    isreachable[i+1][j+k] = 1
                    dp_mat[i][j] += 1

            if projection_matrix[i+1][min(j+offset-1, width-1)] == 0:
                e = j+offset
                while e < width and projection_matrix[i+1][e] == 0:
                    dp_mat[i][j] += 1
                    isreachable[i+1][e] = 1
                    e += 1
            for k in range(1, offset):
                if j < k:
                    break
                if projection_matrix[i+1][j-k] == 0:
                    isreachable[i+1][j-k] = 1
                    dp_mat[i][j] += 1
            if projection_matrix[i+1][max(0, j-offset+1)] == 0:
                e = j-offset
                while e >= 0 and projection_matrix[i+1][e] == 0:
                    dp_mat[i][j] += 1
                    isreachable[i+1][e] = 1
                    e -= 1
            if dp_mat[i][j]:
                continue
            isreachable[i][j] = 0
            backtrack(i, j)
    tmpimg = img.copy()
    part_h = ceil(height/num_zone)
    for i in range(num_zone):
        for j in range(width):
            if isreachable[i][j] == 1 and projection_matrix[i][j] == 0:
                for k in range(part_h*i, min(part_h*(i+1), height)):
                    tmpimg[k][j] = 100


    
    cv2.imshow("tmp", tmpimg)
    cv2.waitKey(0)


for img_name in glob.glob("./test data/*.png"):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    (thresh, img) = cv2.threshold(img, 128,
                                  255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    process(img)
