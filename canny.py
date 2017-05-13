import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt
from imutils import contours
import os
import csv


class CVSExport:
    marks = []

    @classmethod
    def add_mark(cls, filename, mark):
        cls.marks.append((filename, mark))

    @classmethod
    def write_csv(cls):
        with open('./output.csv', 'wb') as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(("FileName", "Mark"))
            for mark in sorted(cls.marks, key=lambda tup: tup[0]):
                w.writerow(mark)


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


file = open("./ModelAnswer.txt", "r")
ModelAnswer = file.readlines()
file.close()

ModelAnswers = {}
bub = {"A": 1, "B" : 2, "C" : 3, "D" : 4}
for answer in ModelAnswer:
        splitLine = answer.split()
        ModelAnswers[splitLine[0]] = splitLine[1]


def grade_15(origin, img, cnts, part):
    grade = 0
    if p == 0:
        s = 1
    elif p == 1:
        s = 16
    else:
        s = 31
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((2, 2), np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    # img = cv2.erode(img, kernel, iterations=2)
    cv2.imshow("Binary", img)
    # img = cv2.dilate(img, kernel3, iterations = 3)
    # img = cv2.erode(img, kernel3, iterations = 1)
    if len(cnts) == 60:
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            corAns = ModelAnswers[str(q + s)]
            multipleSelected = False
            cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
            bubbled = None
            for (j, c) in enumerate(cnts):
                mask = np.zeros(img.shape, dtype="uint8")
                #ellipse = cv2.fitEllipse(c)
                #cv2.ellipse(mask,ellipse,(255),-1)
                cv2.drawContours(mask, [c], -1, (255), -1)

                mask = cv2.erode(mask, kernel2, iterations = 2)
                #cv2.imshow("Mask", mask)
                #cv2.waitKey(0)
                mask = cv2.bitwise_and(img, img , mask=mask)
                #cv2.imshow("Mask", mask)
                #cv2.waitKey(0)
                total = cv2.countNonZero(mask)
                #total = np.count_nonzero((img == [255]).all())
                #print "total", total
                #print mask.shape
                if bubbled is not None and abs(total - bubbled[0] < 35) and total > 255 and bubbled[0] > 255:
                    multipleSelected = True
                        
                if bubbled is None or (total > bubbled[0]):
                    bubbled = (total, j)
            color = (10)
            k = corAns
            if bub[k] == bubbled[1] + 1:
                color = (255)
                if bubbled[0] > 115 and not multipleSelected:
                    grade = grade + 1
            cv2.drawContours(origin, [cnts[bubbled[1]]], -1, color, 3)
        #cv2.imshow("Exam", origin)

        #cv2.waitKey(0)

        return grade
    else:
        return 8

# loading training real marks in a dict
training_real_marks = {}
with open("train_marks.csv", "rb") as trm:
    r = csv.reader(trm, delimiter=",")
    for row in r:
        training_real_marks[row[0]] = row[1]
accErr = 0
img_number = 0
for f in os.listdir("./data/train/original"):
    #f = "S_21_hppscan114.png" FOR EASIER DEBUGGING
    image = cv2.imread("./data/train/original/" + f)
    p = 0
    mark = 0
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
    rot_blurred = rot_blurred[0:height-150,p_l[0]-130:p_r[0]+150]

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
            #cv2.line(rot_blurred,(x1,y1),(x2,y2),(0,255,0),2)

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
                if ((x1>0) and (x1<100)) or ((x1>300) and (x1<450)) or ((x1>600) and (x1<750)):
                    if (min(xs, key=lambda x:abs(x-x1))+20<x1) or (x1<min(xs, key=lambda x:abs(x-x1))-20):
                        xs.append(x1+2)
                        #cv2.line(rot_blurred,(x1,y1),(x2,y2),(0,255,0),2)

    #sorting the lines coordinates
    xs.sort()
    for i in xrange(0,len(xs)-1,1):
        # cropping the image
        crop_img = rot_blurred[0:height, xs[i] :xs[i+1]]
        crop_height, crop_width = crop_img.shape
        crop_img = crop_img[60:680, 100 :crop_width]

        # kernel for openning
        kernel = np.ones((3,3),np.uint8)
        kernel2 =  np.ones((2,2),np.uint8)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

        # Thresholding the image (inverse)
        #th, threshold_crop_img = cv2.threshold(crop_img, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        threshold_crop_img = cv2.adaptiveThreshold(crop_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11 ,2)
        #show_img(threshold_crop_img)

        # closing

        opening_threshold_crop_img = cv2.morphologyEx(threshold_crop_img, cv2.MORPH_CLOSE, kernel3)
        opening_threshold_crop_img = cv2.morphologyEx(opening_threshold_crop_img, cv2.MORPH_OPEN, kernel2)
        #opening_threshold_crop_img = cv2.dilate(opening_threshold_crop_img, kernel3, iterations = 1)
        #show_img(opening_threshold_crop_img)

        # getting contours
        cnts = cv2.findContours(opening_threshold_crop_img.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        questionCnts = []

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if w >= 15 and h >= 15 and ar >= 0.4 and ar <= 1.7:
                questionCnts.append(c)
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        mark = mark + grade_15(crop_img,opening_threshold_crop_img, questionCnts, p)
        p = p + 1
        # color to draw the contours
        color = (0, 255, 255)
        for i in xrange(0,len(questionCnts),1):
            cv2.drawContours(crop_img, [questionCnts[i]], -1, color, 3)

        #name = "./data/saved/" + str(img_number) + ".png"
        #cv2.imwrite(name, crop_img)
        #img_number +=1
        #print len(questionCnts)==60, len(questionCnts)

        #show_img(crop_img)
        #show_img(opening_threshold_crop_img)
    if int(training_real_marks[f]) == mark:
        print("Correct!!")
    else:
        print("HAHAHAHAHA Error, real mark {} Got {} :p el 3yal htes2at xD\n".format(training_real_marks[f], mark))
        accErr = accErr + np.abs(int(training_real_marks[f]) - mark)
    CVSExport.add_mark(f, mark)
print accErr
CVSExport.write_csv()
