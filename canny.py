import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils import contours
import os


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
def grade_15(img):
    grade = 0
    return grade

for f in os.listdir("./data/train/original"):
    image = cv2.imread("./data/train/original/" + f)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=100, param2=40,
                               minRadius=30, maxRadius=50)

    # detected big bold circles
    circles = np.uint16(np.around(circles))

    # should be 2 only
    if len(circles[0, :]) is not 2:
        next

    # getting rotation angle

    ## identifying points
    p_l, p_r = None, None
    if circles[0, :][0][0] < circles[0, :][1][0]:
        p_l = (circles[0, :][0][0], circles[0, :][0][1])
        p_r = (circles[0, :][1][0], circles[0, :][1][1])
    else:
        p_l = (circles[0, :][1][0], circles[0, :][1][1])
        p_r = (circles[0, :][0][0], circles[0, :][0][1])

    ## get mid point between the two centers
    mid_p = ((p_l[0] + p_r[0])/2.0, (p_l[1] + p_r[1])/2.0)

    rot_angle = np.arctan((p_r[1] - mid_p[1]) / (p_r[0] - mid_p[0]))
    rot_angle = rot_angle * 180 / np.pi

    print(f)
    #print(rot_angle)

    rot_blurred = rotate_bound(blurred, rot_angle)

    height, width = rot_blurred.shape
    rot_blurred = rot_blurred[0:height,p_l[0]-130:p_r[0]+150]

    edged = cv2.Canny(rot_blurred, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edged, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=2)
    
    xs = [width]
    ys = [0]

    for l in lines:
        for x1, y1, x2, y2 in l:
            # extracting horizontal lines only
            if (y1 == y2):
                #detecting the main horizontal lines in the image
                if (min(ys, key=lambda x:abs(x-y1))+20<y1) or (y1<min(ys, key=lambda x:abs(x-y1))-20):
                    ys.append(y1+2)

    ys.sort()
    rot_blurred = rot_blurred[ys[len(ys)-2]:ys[len(ys)-1], 0 :width]

    edged = cv2.Canny(rot_blurred, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edged, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=50, maxLineGap=2)
    for l in lines:
        for x1, y1, x2, y2 in l:
            # extracting vertical lines only
            if (x1 == x2):
                #detecting the two main vertical lines in the image
                if (min(xs, key=lambda x:abs(x-x1))+20<x1) or (x1<min(xs, key=lambda x:abs(x-x1))-20):
                    xs.append(x1+2)
            #cv2.line(rot_blurred,(x1,y1),(x2,y2),(0,255,0),2)

    #sorting the lines coordinates
    xs.sort()
    for i in xrange(0,len(xs)-1,1):
        # cropping the image
        crop_img = rot_blurred[0:height, xs[i]+100 :xs[i+1]]


        # kernel for openning
        kernel = np.ones((2,2),np.uint8)  
        kernel2 =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

        # Thresholding the image (inverse)
        #th, threshold_crop_img = cv2.threshold(crop_img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        threshold_crop_img = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11 ,2)
        #show_img(threshold_crop_img)
        

        # closing
        opening_threshold_crop_img = cv2.morphologyEx(threshold_crop_img, cv2.MORPH_CLOSE, kernel2)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_OPEN, kernel)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_CLOSE, kernel2)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_OPEN, kernel)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_CLOSE, kernel2)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_OPEN, kernel)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_CLOSE, kernel2)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_CLOSE, kernel2)
        #show_img(opening_threshold_crop_img)

        # getting contours
        cnts = cv2.findContours(opening_threshold_crop_img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        questionCnts = []
         
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if w >= 15 and h >= 13 and ar >= 0.7 and ar <= 1.5:
                questionCnts.append(c)
        
        # color to draw the contours
        color = (0, 255, 255)
        for i in xrange(0,len(questionCnts)-1,1):
            cv2.drawContours(crop_img, [questionCnts[i]], -1, color, 3)            

        show_img(crop_img)
        show_img(opening_threshold_crop_img)