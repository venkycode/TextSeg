import cv2
import sys
import numpy as np
import os
import heapq
import random
from math import ceil
import glob

num_zone=80
height=0
width=0
offset=10
dp_mat=list()
isreachable=list()
projection_matrix=list()
header_offset = 0.2
name=str()
cnt=0
source_distance=5
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
    if not end_position:
        end_position=height
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

def backtrack(i,j):
    #part_h= ceil(height/num_zone)
    if i==0:
        return 
    rl=j  #right limit
    ll=j  #left limit
    for y in range(j,width):
        if projection_matrix[i][y]:
            break
        rl = y
    for y in range(j, 0, -1):
        if projection_matrix[i][y]:
                break
        ll=y

    ll=max(0, ll-(offset-1))
    rl=min(width-1, rl+(offset-1))
    for y in range(ll,rl+1):
        if projection_matrix[i-1][y]==0:
            dp_mat[i-1][y]-=1
    for y in range(ll,rl+1):
        if dp_mat[i-1][y]==0 and projection_matrix[i-1][y]==0 and isreachable[i-1][y]==1:
            isreachable[i-1][y]=0
            backtrack(i-1,y)
    return

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

def primary_segmentation(img):
    vertical_projection = getVerticalProjectionProfile(img)
    min_density = min(vertical_projection)
    max_density = max(vertical_projection)

    #print(vertical_projection)
    cutoff_density = (max_density-min_density)*0.1 + min_density

    source_segment_points = list()

    width = img.shape[1]
    height = img.shape[0]

    for i in range(width):
        if vertical_projection[i] <= cutoff_density:
            source_segment_points.append((i, vertical_projection[i]))

    '''
    print(source_segment_points)'''
    source_segment_points = processSources(source_segment_points, source_distance)
    source_segment_points
    #source_segment_points= processSources(source_segment_points,40)
    #source_segment_points= processSources(source_segment_points,60)

    # checker code

    tmpimg= img.copy()
    for i in source_segment_points:
        for j in range(height):
            tmpimg[j][i] = 0
    #show_image(tmpimg)

    return (source_segment_points,cutoff_density,tmpimg)

def find_path(source):
    sz=len(projection_matrix)
    path=list()
    path.append(source)
    cur=0
    off=20
    last=source
    while(cur+1<sz):
        flag=1
        for i in range(off):
            if isreachable[cur+1][min(last+i,width-1)]:
                path.append(min(last+i,width-1))
                last=min(last+i,width-1)
                flag=0
                break
            if isreachable[cur+1][max(0,last-i)]:
                path.append(max(0,last-i))
                flag=0
                last=max(0,last-i)
                break
        if abs(last-source)>20:
            flag=1
        if flag:
            break
        cur+=1
    if len(path)==sz:
        return path
    else:
        l=list()
        return l
    

def process(img):
    #remove_header(img)
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
                if k+j>=width:break
                if projection_matrix[i+1][j+k]==0:
                    isreachable[i+1][j+k]=1
                    dp_mat[i][j]+=1
                
            if projection_matrix[i+1][min(j+offset-1,width-1)]==0:
                e=j+offset
                while e<width and projection_matrix[i+1][e]==0:
                    dp_mat[i][j]+=1
                    isreachable[i+1][e]=1
                    e+=1
            for k in range(1,offset):
                if j<k:break
                if projection_matrix[i+1][j-k]==0:
                    isreachable[i+1][j-k]=1
                    dp_mat[i][j]+=1
            if projection_matrix[i+1][max(0,j-offset+1)]==0:
                e=j-offset
                while e>=0 and projection_matrix[i+1][e]==0:
                    dp_mat[i][j]+=1
                    isreachable[i+1][e]=1
                    e-=1
            if dp_mat[i][j]:continue
            isreachable[i][j]=0
            backtrack(i,j)
    # tmpimg=img.copy()
    part_h= ceil(height/num_zone)
    # for i in range(num_zone):
    #     for j in range(width):
    #         if isreachable[i][j]==1 and projection_matrix[i][j]==0:
    #             for k in range(part_h*i, min(part_h*(i+1), height)):
    #                     tmpimg[k][j]=100
    (sources,cutoff_density,tmp)=primary_segmentation(img)
    vert_pro=getVerticalProjectionProfile(img,0,height)
    #print(sources)
    sz=len(sources)
    #inter_sources=list()
    for i in range(1,sz):
        if sources[i]-sources[i-1]>140:
            st=sources[i-1]+40
            while(st< sources[i]):
                if(vert_pro[st] ):
                    sources.append(st)
                st+=40
    #print(sources)
    sources.sort()
    #print(sources)
    tmpimg=img.copy()
    for i in sources:
        for j in range(height):
            tmpimg[j][i]=120
    # cv2.imshow("tmp",tmpimg)
    # cv2.waitKey(0)
    path=list()
    for i in sources:
        path.append(find_path(i))
    for i in range(len( path)):
        for j in range(len(path[i])):
            for x in range(part_h*j,min(height,part_h*(j+1))):
                img[x][path[i][j]]=0

    global cnt
    cv2.imshow("tmp1",img)
    cv2.waitKey(1000)
    cv2.imwrite("./resori/seg_"+str(cnt)+".jpg",img)
    print("./resori/seg_"+str(cnt)+".jpg")
    cnt+=1

    
    # cv2.imshow("tmp", tmpimg)
    # cv2.waitKey(0)


for img_name in glob.glob("./test/oriya/oriya/*.png"):
    #name=img_name
    if(cnt==100):
        sys.exit()
    img= cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    (thresh, img) = cv2.threshold(img, 128,
                                  255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    process(img)