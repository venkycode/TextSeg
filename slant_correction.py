import cv2
import sys
import numpy as np
from math import ceil

input_img = "test data/im22.png"
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

def find_si(i, pi):
    slope=(pi-i)/(N-1)
    ans=0
    cur=0
    for j in range(N):
        y=i+slope*(j-i)
        fp=y-int(y)
        y=int(y)
        if fp>=0.5:y+=1
        if img[j][y]==0 : cur+=1
        else :
            ans=max(ans,cur)
            cur=0
    ans=max(ans,cur)
    return ans

def find_gamma(i, pi, prev_pi):
    if pi==prev_pi+1:return 0
    slope=(pi-i)/(N-1)
    ans=0
    for j in range(N):
        y=i+slope*(j-i)
        fp=y-int(y)
        y=int(y)
        if fp>=0.5:y+=1
        if img[j][y]==0 :ans+=1
    return -ans


def find_ci(i, pi):
    # use trial and error 

def find_f(i, pi, prev_pi):
    return ws*find_si(i,pi)+wc*find_ci(i,pi)+wn*find_gamma(i,pi,prev_pi)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img = cropROI(img)
N = img.shape[0]
M = img.shape[1]
W = 2*N
print(img)
ws=0.56
wc=0.26
wn=0.18
g = [[0 for i in range(M)] for j in range(M)]
b = [[0 for i in range(M)] for j in range(M)]
for pi in range(W+1):g[0][pi]=ws*find_si(0,i)+wc*find_ci(0,i)
for i in range(1,M):
    for pi in range(max(i-W,0),min(i+W+1,M)):
        for k in range(2):
            cur=g[i-1][pi-k]+find_f(i,pi,pi-k)
            if cur>g[i][pi]:
                g[i][pi]=cur
                b[i][pi]=pi-k

opt=[]
for i in range(M):opt.append(0)
max_b=0
for i in range(M-W,M):
    if g[M-1][i]>max_b:opt[M-1]=i
for i in range(M-1,1):opt[i-1]=b[i][opt[i]]




cv2.imshow(input_img,img)
cv2.waitKey(0)
