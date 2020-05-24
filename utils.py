import math
import cv2
import numpy as np

def get_canny(img):
    """
    Input: take a raw image
    Output: detect edges of img using canny
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(img):
    """
    Input: get a edge img, vertices as the limit of ROI
    Output: A mask of img which only contains ROI
    """
    height = img.shape[0]
    polygon = np.array([[(100,height), (1100, height), (600,150)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    mask_img = cv2.bitwise_and(img, mask)
    return mask_img

def combine_line_img(img, line_img):
    return cv2.addWeighted(img, 0.8, line_img, 1, 1)

def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_img, (x1, y1),(x2, y2), (255,0,0), 10)
    return line_img

def get_houghlines(mask):
    """
    Input: Canny mask of an image
    Ouput: Lines detector
    """
    return cv2.HoughLinesP(mask, 2, np.pi/180,100, np.array([]), minLineLength=40, maxLineGap=5)

def make_coordinates(img, line_params):
    slope, intercept = line_params
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope <0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_avg)
    right_line = make_coordinates(img, right_fit_avg)
    return np.array([left_line, right_line])


