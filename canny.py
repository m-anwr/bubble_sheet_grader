import cv2
import numpy as np
from matplotlib import pyplot as plt
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
    rot_angle = None
    if p_l[1] > mid_p[1]:
        rot_angle = np.arctan((p_l[1] - mid_p[1]) / (mid_p[0] - p_l[0]))
    else:
        rot_angle = np.arctan((p_r[1] - mid_p[1]) / (p_r[0] - mid_p[0]))

    rot_angle = rot_angle * 180 / np.pi
    print(f)
    print(rot_angle)

    rot_blurred = rotate_bound(blurred, rot_angle)

    rot_blurred = rot_blurred[p_l[1]-850:p_r[1]-100,
                              p_l[0]-130:p_r[0]+130]
    # show_img(rot_blurred)

    edged = cv2.Canny(rot_blurred, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edged, rho=1, theta=np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=2)

    height, width = rot_blurred.shape
    xs = [width]


    # print(len(lines))

    for l in lines:
        for x1, y1, x2, y2 in l:
            if (x1 == x2):
                if (min(xs, key=lambda x:abs(x-x1))+20<x1) or (x1<min(xs, key=lambda x:abs(x-x1))-20):
                    xs.append(x1+2)
            # cv2.line(rot_blurred,(x1,y1),(x2,y2),(0,255,0),2)

    # show_img(rot_blurred)
    xs.sort()
    for i in xrange(0,len(xs)-1,1):
        print xs[i],xs[i+1]
        crop_img = rot_blurred[0:height, xs[i]:xs[i+1]]

    kernel = np.ones((5,5),np.uint8)      

    threshold_crop_img = cv2.threshold(crop_img, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    opening_threshold_crop_img = cv2.morphologyEx(threshold_crop_img, cv2.MORPH_OPEN, kernel)

    show_img(opening_threshold_crop_img)

