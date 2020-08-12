import cv2
import sys
import numpy as np
from math import ceil
import math

input_img = "./input"
img = cv2.imread(input_img+".png")
 
 
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
            
            if(i): start_x = i-1
            else: start_x=i
            break
    for i in range(width-1, 0, -1):
        if verticle_projection[i]:
            if(width-1-i):end_x = i+1
            else: end_x= width-1
            break
    for i in range(height):
        if horizontal_projection[i]:
            if(i):start_y = i-1
            else: start_y=i
            break
    for i in range(height-1, 0, -1):
        if horizontal_projection[i]:
            if(height-1-i):end_y = i+1
            else: height=i
            break
    image = image[start_y:end_y, start_x:end_x]
    return image
 
 
def find_si(i, pi):
    slope = (pi-i)/(N-1)
    ans = 0
    cur = 0
    for j in range(N):
        y = i+slope*j
        fp = y-math.floor(y)
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
    if ans<N/3:ans=0
    return ((ans/N)*64) # this change is done so that it adapts irresptive of resolution change
 
 
def find_gamma(i, pi, prev_pi):
    if pi == prev_pi+1:
        return 0
    if i==W: return 0
    slope = (pi-i)/(N-1)
    ans = 0
    for j in range(N):
        y = i+slope*j
        fp = y-int(y)
        y = int(y)
        if fp >= 0.5:
            y += 1
        if(y >= M):
            break
        if img[j][y] == 0:
            ans += 1
    return -ans
 
def find_f(i, pi, prev_pi):
    return ws*find_si(i, pi)+wn*find_gamma(i, pi, prev_pi)
 
cur_h = img.shape[0]
ratio =500/cur_h
dim = (int(img.shape[1]*ratio), int(img.shape[0]*ratio))
img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
(thresh, img) = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
cv2.imwrite(input_img+"upscaled.png",img)
 
new_img=cv2.imread(input_img+"upscaled.png")
new_img=cv2.bitwise_not(new_img)
 
new_img = cv2.ximgproc.thinning(cv2.cvtColor(new_img,cv2.COLOR_RGB2GRAY))
cv2.imwrite(input_img+"thinned_img.png",new_img);
(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
ratio =128/img.shape[0]
dim = (int(img.shape[1]*ratio), int(img.shape[0]*ratio))
img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
N = img.shape[0]
M = img.shape[1]
print(N,M)
W = 2*N
img= cv2.copyMakeBorder(img,0,0,W,W,cv2.BORDER_CONSTANT,value=255)

N = img.shape[0]
M = img.shape[1]
W = 2*N
print(N,M)
ws = 0.75
wc = 0
wn = 0.25
sig = 11* (N/64)
 
#sigma is large here because the resolution of image is high , we can adjust it to a lower value by lowering resoltion

g = [[0 for i in range(M)] for j in range(M)]
b = [[0 for i in range(M)] for j in range(M)]
 
for pi in range(min(2*W+1,M)):g[W][pi]=ws*find_si(W,pi)+wc*find_ci(W,pi)
for i in range(1+W,M):
    print(i)
    for pi in range(max(i-W,0),min(i+W+1,M)):
        g[i][pi]=-100000000000000
        for k in range(3):
            cur=g[i-1][pi-k]+find_f(i,pi,pi-k)
            #print(cur,end=" ")
            if cur>g[i][pi]:
                #print(i,pi,cur)
                g[i][pi]=cur
                b[i][pi]=pi-k
            elif cur==g[i][pi]:
                b[i][pi]= pi-1
 
opt=[]
for i in range(M):opt.append(0)
for i in range(W): opt[i]=i
opt[M-1]=M-1
print(opt[M-W-1],end=" ")
for i in range(M-1,W,-1):
    opt[i-1]=b[i][opt[i]]
    print(opt[i-1],end=" ")

new_opt=[]
new_height= new_img.shape[0]
new_img=cv2.copyMakeBorder(new_img,0,0,new_height*2,new_height*2,cv2.BORDER_CONSTANT,value=255)
new_width= new_img.shape[1]
for i in range(new_width):new_opt.append(0)
ratio=new_img.shape[0]/img.shape[0]
new_cur_i=0

for i in range(M):
    new_slope= (opt[i]-i)/(N-1)
    
    while(new_cur_i<new_width):
        if((new_cur_i+1)/(i+1))>=ratio:
            break
        new_opt[new_cur_i]= (new_slope*(new_height-1) + new_cur_i)
        new_cur_i +=1 
    
 
final_img= new_img.copy()
 
for x in range(2*new_height,new_img.shape[1]):
    for y in range(new_img.shape[0]):
        i=x
        pi=new_opt[i]
        xx= i+((pi-i)/(new_height-1))*y
        fp=xx-int(xx)
        xx=int(xx)
        if fp>=0.5:xx+=1
        if xx<new_img.shape[1] and xx>=0:
            final_img[y][x]=new_img[y][xx]
        else:
            print(x,y,xx)
print(new_img.shape)

cv2.imwrite("result1.jpg",final_img)
cv2.imwrite("inter.jpg",img)
 