# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import imageio
from moviepy.editor import *
from IPython.display import HTML




def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def line_extrapolation(lines, line_image,upper_ROI):

    left_lines=np.empty((0,2))
    right_lines=np.empty((0,2))

    for line in lines:
        for x1,y1,x2,y2 in line:
            slope=(y2-y1)/(x2-x1)
            b=y1-(slope*x1)
            if -0.4 >= slope >= -0.9:
                left_lines=np.vstack((left_lines,np.array([slope,b])))
            elif 0.4 <= slope <= 0.9:
                right_lines=np.vstack((right_lines,np.array([slope,b])))
            else:
                print("Outlier Line Fragment- Discarded")
                continue

    points=[left_lines, right_lines]

    for side in points:
        try:
            m,b=np.mean(side, axis=0)
            #----------P_down------------
            y1=line_image.shape[0]
            x1=(y1-b)/m
            #----------P_up------------
            y2=upper_ROI
            x2=(y2-b)/m

            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), color=[0, 255, 0], thickness=5)
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!SHIT FRAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

#list images in folder
#os.listdir("test_images/")
#reading in an image
#image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

def select_white_yellow(image):
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def process_image(pic):

    #raw = mpimg.imread(pic)
    image=np.copy(pic)

    low_threshold=50
    high_threshold=150
    rho=1
    theta=np.pi/180
    threshold=20
    min_line_len=20
    max_line_gap=20
    kernel_size=5

    #----------FOV----------
    vertices=np.array([[[
        [0,image.shape[0]],
        [image.shape[1]/2-120,0.6*image.shape[0]],
        [image.shape[1]/2+120,0.6*image.shape[0]],
        [image.shape[1],image.shape[0]]]]],dtype=np.int32)

    upper_ROI=0.7*image.shape[0]

    selected=select_white_yellow(image)

    #plt.imshow(gray, cmap='gray')  # this colormap will display in black / white
    #plt.show()

    gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY)
    gray_select=cv2.inRange(gray,0,150)
    #selected=cv2.inRange(gray,low_threshold, high_threshold)
    #plt.imshow(selected)
    #plt.show()

    blur_res=gaussian_blur(gray_select, kernel_size)
    edges=canny(blur_res, low_threshold, high_threshold)
    #plt.imshow(edges,cmap='Greys_r')
    #plt.show()

    masked = region_of_interest(edges, vertices)
    #plt.imshow(masked)
    #plt.show()

    line_pic,lines=hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)
    #plt.imshow(line_pic,cmap='Greys_r')
    #plt.show()

    line_image = np.copy(image)*0
    line_extrapolation(lines,line_image,upper_ROI)
    #plt.imshow(line_image)
    #plt.show()

    combo = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    #plt.imshow(combo)
    #plt.show()

    return combo

    #image_copy=grayscale(image_copy)
    #image_copy=gaussian_blur(image_copy, kernel_size)
    #image_copy=canny(image_copy, low_threshold, high_threshold)
    #image_copy=hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap)

if __name__=="__main__":
    white_output = 'test_videos_output/challenge.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    clip = VideoFileClip("test_videos/challenge.mp4")
    #clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    #processed_clip=process_image(clip1)
    clip=clip.fl_image(process_image)
    #white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    clip.write_videofile(white_output, audio=False)






















