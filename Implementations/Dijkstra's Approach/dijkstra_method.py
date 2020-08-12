import cv2
import sys
import numpy as np
import imutils
import os
import heapq
import random
from math import ceil
import glob

header_offset = 0.2
cutoff_factor = 0.2  # adjustable
source_distance=10
offset = 30
shortest_path_cutoff=0.35

def show_image(image, window_name='tmp'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


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

def getVerticalProjectionProfile(image):
    height= image.shape[0]
    width= image.shape[1]
    x = [[0 for i in range(width)] for j in range(height)]
    for i in range(height):
        for j in range(width):
            if image[i][j] == 0:
                x[i][j] = 1

    vertical_projection = np.sum(x, axis=0)
    return vertical_projection

def remove_header(img): #removes header in from devanagari like scripts
    horizontal_projection = getHorizontalProjectionProfile(img)
    mx = max(horizontal_projection)
    for i in range(img.shape[0]):
        if(abs(mx-horizontal_projection[i]) <= mx*header_offset):
            for j in range(img.shape[1]):
                img[i][j] = 255



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
    
    cutoff_density = (max_density-min_density)*cutoff_factor + min_density

    source_segment_points = list()

    width = img.shape[1]
    height = img.shape[0]

    for i in range(width):
        if vertical_projection[i] <= cutoff_density:
            source_segment_points.append((i, vertical_projection[i]))

    source_segment_points = processSources(source_segment_points, source_distance)
    source_segment_points
    tmpimg= img.copy()
    for i in source_segment_points:
        for j in range(height):
            tmpimg[j][i] = 0
    return (source_segment_points,cutoff_density,tmpimg)


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
    inf = 100000000000000
    height = image.shape[0]
    width = image.shape[1]
    miny = max(0, source-offset)
    maxy = min(width, source+offset)
    start = (0, 0, 0, source)
    queue = []
    queue.append(start)
    heapq.heapify(queue) #priority queue
    
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
        if x == height-1:
            destination = (x, y)
            path_distance = d
            break
        visited[x][y] = 1

        if y-1 >= miny:
            if visited[x][y-1] == 0:
                new_d = d + weights(image[x][y], image[x][y-1])
                seed = random.randint(0, 100) # randomisation to avoid unwanted preference to particular direction
                if dist[x][y-1] > new_d:
                    heapq.heappush(queue, (new_d, seed, x, y-1)) #custom tuple
                    dist[x][y-1] = new_d
                    parent[x][y-1] = (x, y)

        if y+1 < maxy:
            if visited[x][y+1] == 0:
                new_d = d + weights(image[x][y], image[x][y+1])
                seed = random.randint(0, 100)
                if dist[x][y+1] > new_d:
                    heapq.heappush(queue, (new_d, seed, x, y+1))
                    dist[x][y+1] = new_d
                    parent[x][y+1] = (x, y)

        if x-1 >= 0:
            if visited[x-1][y] == 0:
                new_d = d + weights(image[x][y], image[x-1][y])
                seed = random.randint(0, 100)
                if dist[x-1][y] > new_d:
                    heapq.heappush(queue, (new_d, seed, x-1, y))
                    dist[x-1][y] = new_d
                    parent[x-1][y] = (x, y)

        if x+1 < height:
            if visited[x+1][y] == 0:
                seed = random.randint(0, 100)
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

    cutoff_distance = shortest_path_cutoff*cutoff
    if path_distance <= cutoff_distance:
        return path
    return []

def djikstra_segmentation(img,sources,cutoff_density):
    for source in sources:
        path = dijikstra(img, source, cutoff_density)
        for p in path:
            img[p[0]][p[1]] = 0

def process_image(img):
    remove_header(img)
    
    (sources,cutoff_density,tmp)=primary_segmentation(img) 
    final_img=[[0 for i in range(2*img.shape[1]+20)] for j in range(img.shape[0])]
    final_img=np.float32(final_img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            final_img[i][j]=tmp[i][j]
    
    for i in range(img.shape[0]):
        for j in  range(20):
            final_img[i][img.shape[1]+j]=0

    djikstra_segmentation(img,sources,cutoff_density)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            final_img[i][j+img.shape[1]+20]=img[i][j]
    
    return final_img

def main():
    cnt=0
    for image in glob.glob('./test_words/*.jpg'): # replace with path to folder containing words to segment
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, img) = cv2.threshold(img, 128,
                                      255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #binarization
        final=process_image(img)
        file='./segmented_hindi/seg_'+str(cnt)+'.jpg'
        cnt+=1
        print(file)
        cv2.imwrite(file,final)

main()
