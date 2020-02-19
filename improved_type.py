import cv2
import sys
import numpy as np
import imutils
import os
import heapq
import random
from math import ceil

input_img = "im1.png"
img = cv2.imread(input_img)


def getVerticalProjectionProfile(image, header_position=0):
    height = image.shape[0]
    width = image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]

    for i in range(header_position+1, height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    vertical_projection = np.sum(x, axis=0)

    return vertical_projection


def getHorizontalProjectionProfile(image):

        # Convert black spots to ones
    height = image.shape[0]
    width = image.shape[1]
    x = [[0 for i in range(image.shape[1])] for j in range(image.shape[0])]

    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
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


def processSources(listSources, dif_factor):

    new_list = []
    new_list.append(listSources[0][0])
    sz = len(listSources)
    min_tuple = (-1, 99999999999999999)
    for i in range(sz):
        if listSources[i][1] < min_tuple[1]:
            min_tuple = listSources[i]
        if i != sz-1 and listSources[i+1][0]-listSources[i][0] >= dif_factor:
            new_list.append(min_tuple[0])
            min_tuple = (-1, 99999999999999999)
    new_list.append(listSources[sz-1][0])

    return new_list


def weights(pixel_a,  pixel_b):
    #print(pixel_a, pixel_b)
    sum = int(bool(pixel_a)) + int(bool(pixel_b))
    if sum == 1:
        return 1
    elif sum == 2:
        return 0.001
    else:
        return 1


def dijikstra(image, source, cutoff):
    # print(source)
    offset = 40
    inf = 100000000000000
    height = image.shape[0]
    width = image.shape[1]
    miny = max(0, source-offset)
    maxy = min(width, source+offset)
    start = (0, 0, 0, source)
    queue = []
    queue.append(start)
    heapq.heapify(queue)
    visited = [[0 for i in range(image.shape[1])]
               for j in range(image.shape[0])]
    dist = [[inf for i in range(image.shape[1])]
            for j in range(image.shape[0])]
    parent = [[(-1, -1) for i in range(image.shape[1])]
              for j in range(image.shape[0])]
    dist[start[0]][start[1]] = 0
    destination = (-1, -1)
    path_distance = -1
    while queue:
        node = heapq.heappop(queue)
        x = node[2]
        y = node[3]
        d = node[0]
        #print("node ",node)
        if x == height-1:
            destination = (x, y)
            path_distance = d
            break
        visited[x][y] = 1

        if y-1 >= miny:
            if visited[x][y-1] == 0:
                new_d = d + weights(image[x][y], image[x][y-1])
                seed = random.randint(0, 100)
                # seed=0
                if dist[x][y-1] > new_d:
                    heapq.heappush(queue, (new_d, seed, x, y-1))
                    dist[x][y-1] = new_d
                    parent[x][y-1] = (x, y)

        if y+1 < maxy:
            if visited[x][y+1] == 0:
                new_d = d + weights(image[x][y], image[x][y+1])
                seed = random.randint(0, 100)
                # seed=0
                if dist[x][y+1] > new_d:
                    heapq.heappush(queue, (new_d, seed, x, y+1))
                    dist[x][y+1] = new_d
                    parent[x][y+1] = (x, y)

        if x-1 >= 0:
            if visited[x-1][y] == 0:
                new_d = d + weights(image[x][y], image[x-1][y])
                seed = random.randint(0, 100)
                # seed=0
                if dist[x-1][y] > new_d:
                    heapq.heappush(queue, (new_d, seed, x-1, y))
                    dist[x-1][y] = new_d
                    parent[x-1][y] = (x, y)

        if x+1 < height:
            if visited[x+1][y] == 0:
                seed = random.randint(0, 100)
                # seed=0
                new_d = d + weights(image[x][y], image[x+1][y])
                if dist[x+1][y] > new_d:
                    heapq.heappush(queue, (new_d, seed, x+1, y))
                    dist[x+1][y] = new_d
                    parent[x+1][y] = (x, y)

    path = []
    cur = destination
    source_tuple = (0, source)
    while cur != source_tuple:
        path.append(cur)
        cur = parent[cur[0]][cur[1]]

    path.append(source_tuple)

    cutoff_distance = 0.5*cutoff
    if path_distance <= cutoff_distance:
        return path
    return []


def processForConnectedLetters(source_segment_points,img):
    cur=0 
    mn=1000000000
    new_points=list()
    for i in source_segment_points:
        if(i-cur):
            mn=min(mn,i-cur)
        cur=i
    #if(img.shape[1]-cur):
        #mn=min(img.shape[1]-cur,mn)
    
    cur=0
    print("mn ",mn)
    
    for i in source_segment_points:
        if (i-cur)>=1.5*mn:
            new_points.append(ceil(cur+(i-cur)*0.5))
        cur=i
    for i in new_points:
        source_segment_points.append(i)
    return source_segment_points

if os.path.isfile("header_removed"+input_img) == 0:
    print("here")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(img, 128,
                                  255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height = img.shape[0]
    width = img.shape[1]
    pro = getHorizontalProjectionProfile(img)
    mx=max(pro)
    header_position1 = -1
    header_position2 = -1
    for i in range(height):
        if(abs(mx-pro[i]) <= mx*0.40):
            if header_position1 == -1:
                header_position1 = i
            header_position2 = i
            for j in range(width):
                img[i][j] = 255
    #correctHeader(img,header_position1,header_position2)
    for i in range(max(header_position1-5,0),header_position1):
        for j in range(img.shape[1]):
            img[i][j]=255
    cv2.imshow('output.png', img)
    cv2.waitKey(0)
    cv2.imwrite("header_removed"+input_img, img)


img = cv2.imread("header_removed"+input_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print(img)
img = cropROI(img)
# cv2.imshow("tmp.jpg",img)
# cv2.waitKey(0)
cv2.imwrite("cropped_"+input_img, img)


vertical_projection = getVerticalProjectionProfile(img)

min_density = min(vertical_projection)
max_density = max(vertical_projection)

cutoff_factor = 0.2  # adjustable

cutoff_density = (max_density-min_density)*cutoff_factor + min_density

source_segment_points = list()

width = img.shape[1]
height = img.shape[0]

for i in range(width):
    if vertical_projection[i] <= cutoff_density:
        source_segment_points.append((i, vertical_projection[i]))

'''
print(source_segment_points)'''
source_segment_points = processSources(source_segment_points, 20)
print(source_segment_points)
#source_segment_points= processForConnectedLetters(source_segment_points,img)
#print(source_segment_points)
#source_segment_points= processSources(source_segment_points,40)
#source_segment_points= processSources(source_segment_points,60)

# checker code

tmpimg = img.copy()
for i in source_segment_points:
    for j in range(height):
        #print(i,j)
        tmpimg[j][i] = 155

cv2.imwrite("basic_seg"+input_img, tmpimg)

#-------------------#


tmpimg = img.copy()

for source in source_segment_points:
    path = dijikstra(img, source, cutoff_density)
    for p in path:
        tmpimg[p[0]][p[1]] = 150

cv2.imwrite("final_seg_"+input_img, tmpimg)
