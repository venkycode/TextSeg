import cv2
import sys
import numpy as np
from math import ceil
import math

input_img = "./testimg.jpeg"
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
    slope = (pi-i)/(N-1)
    ans = 0
    cur = 0
    for j in range(N):
        y = i+slope*(j-i)
        fp = y-int(y)
        y = int(y)
        if fp >= 0.5:
            y += 1
        if(y >= M):
            break
        if img[j][y] == 0:
            cur += 1
        else:
            ans = max(ans, cur)
            cur = 0
    ans = max(ans, cur)
    return ans


def find_gamma(i, pi, prev_pi):
    if pi == prev_pi+1:
        return 0
    slope = (pi-i)/(N-1)
    ans = 0
    for j in range(N):
        y = i+slope*(j-i)
        fp = y-int(y)
        y = int(y)
        if fp >= 0.5:
            y += 1
        if(y >= M):
            break
        if img[j][y] == 0:
            ans += 1
    return -ans


def avg_pi(centr, i):
    # print(type(N))

    new_img = img[:, int(max(centr - sig/2, 0)): int(min(M, centr + sig/2))]
    new_img = 255 - new_img
    contours, _ = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # cv2.drawContours(new_img, contours, -1, (0, 0, 0), thickness=-1)
    # cv2.imshow("img", new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    one_part = 360/16
    #print(contours[0])
    cnt = np.zeros(8, dtype=int)
    for cont in contours:
        first = cont[0][0]
        l = len(cont)
        for i in range(1, l):
            # print(cont[i][0])
            angle = (math.atan2(cont[i][0][1]-first[1],cont[i][0][0]-first[0])) * (180/math.pi)
            if(angle < 0):
                angle += 180
            #print(angle)
            cnt[int(math.floor(angle/one_part)) % 8] += 1

    out = i + N*(((2*cnt[1]+2*cnt[2]+cnt[3])-(cnt[5]+2*cnt[6]+2*cnt[7])) /
                 ((cnt[1]+2*cnt[2]+2*cnt[3])+2*cnt[4]+(2*cnt[5]+2*cnt[6]+cnt[7])))
    #print(out, i)
    return out


def find_ci(i, pi):

    # use trial and error
    return -1 * abs(avg_pi(min(i, pi)+abs(i-pi)/2, i) - pi)


def find_f(i, pi, prev_pi):
    return ws*find_si(i, pi)+wc*find_ci(i, pi)+wn*find_gamma(i, pi, prev_pi)


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
img = cropROI(img)
cur_h = img.shape[0]
ratio = 128/cur_h
dim = (int(img.shape[1]*ratio), int(img.shape[0]*ratio))
img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow("imm", img)
cv2.waitKey(0)
N = img.shape[0]
M = img.shape[1]
print(N, M)
W = 2*N
print(img)
ws = 0.56
wc = 0.26
wn = 0.18
sig = 22
new_img= img.copy()
new_img= 255- new_img
contours, _ = cv2.findContours(new_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.imshow("img", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.drawContours(new_img, contours, -1, (0, 0, 0), thickness=0)
cv2.imshow("img", new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("cont.jpg",new_img)
#sigma is large here because the resolution of image is high , we can adjust it to a lower value by lowering resoltion
#avg_pi(60)
g = [[0 for i in range(M)] for j in range(M)]
b = [[0 for i in range(M)] for j in range(M)]
for pi in range(min(W+1,M)):g[0][pi]=ws*find_si(0,pi)+wc*find_ci(0,pi)
for i in range(1,M):
    print(i)
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
new_img=img.copy()
for x in range(M):
    for y in range(N):
        i=x
        pi=opt[i]
        xx= int( math.floor(((pi-i)*(y-N+1)/(N-1))+pi))
        #print(x,y)
        new_img[y][x]=img[y][xx]
print(new_img.shape)
print(opt)
cv2.imshow("new_img",new_img)
cv2.imwrite("result.jpg",new_img)
cv2.imwrite("inter.jpg",img)
cv2.waitKey(0)
