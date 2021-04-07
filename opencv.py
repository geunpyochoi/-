#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')

image = mping.imread('img213.PNG')

plt.figure(figsize=(10,8))
print(type(image),image.shape)
plt.imshow(image)
plt.show()


# In[2]:


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = grayscale(image)
plt.figure(figsize=(10,8))
plt.imshow(gray,cmap='gray')
plt.show()


# In[4]:


def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img,(kernel_size,kernel_size,),0)

kernel_size = 5
blur_gray = gaussian_blur(gray,kernel_size)

plt.figure(figsize=(10,8))
plt.imshow(blur_gray,cmap='gray')
plt.show()


# In[5]:


def canny(img,low_threshold, high_threshold):
    return cv2.Canny(img,low_threshold,high_threshold)
low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold,high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges,cmap='gray')
plt.show()


# In[6]:


import numpy as np

mask = np.zeros_like(image)

plt.figure(figsize=(10,8))
plt.imshow(mask,cmap='gray')
plt.show()


# In[7]:


if len(image.shape) > 2:
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255
    
imshape = image.shape
print(imshape)

vertices = np.array([[(100,imshape[0]),
                     (450,320),
                     (550,320),
                     (imshape[1]-20,imshape[0])]], dtype=np.int32)
cv2.fillPoly(mask, vertices, ignore_mask_color)

plt.figure(figsize=(10,8))
plt.imshow(mask,cmap='gray')
plt.show()


# In[8]:


def region_of_interest(img,vertices):
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shaep[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image


# In[9]:


imshape = img.shape
vertices = np.array([[(100,imshape[0]),
                     (450,320),
                     (550,320),
                     (imshape[1]-20,imshape[0])]], dtype=np.int32)
mask = region_of_interest(edges,vertices)

plt.figure(figsize = (10,8))
plt.imshow(mask,cmap = 'gray')
plt.show()


# In[11]:


def draw_lines(img,lines,color=[255,0,0],  thickness = 5):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)
def hough_lines(img,rho,theta,threshold,min_line_len,max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    draw_lines(line_img,lines)
    return line_img
rho = 2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

lines = hough_lines(mask,rho,theta,threshold,
                   min_line_len,max_line_gap)
plt.figure(figsize=(10,8))
plt.imshow(lines,cmap='gray')
plt.show()


# In[ ]:


def weighted_img(img,initial_img, α=0.8,β=1.,λ=0.):
    return cv2.addWeighted(initial_img,α,img,β,λ)

lines_edges = weighted_img(lines, img,α=0.8,β=1.,λ=0.)
plt.figure(figsize=(10,8))
plt.imshow(lines_edges)
plt.show()

