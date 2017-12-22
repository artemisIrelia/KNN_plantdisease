import cv2
import imutils
import numpy as np


def disease(filename):
    image = cv2.imread(filename)
    ratio = image.shape[0] / 300.0
    orig = image.copy()

    image = imutils.resize(image, height=300)
    orig = image.copy()

    # Z = image.reshape((-1, 3))
    # Z = np.float32(Z)
    # # define criteria, number of clusters(K) and apply kmeans()
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # K = 6
    # ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # res2 = res.reshape((image.shape))
    #
    # cv2.imshow('res2', res2)
    #
    # image = res2.copy()


    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 11, 15, 17)
    # cv2.imshow("cvt color", gray)
    # ret, edged = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", edged)
    edged = cv2.Canny(image, 30, 200)
    # cv2.imshow("grayed", edged)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    circle = 0
    square = 0
    cntr_count = len(cnts)
    square_countour_list = []
    countour_list = []
    rect_areas = []
    circ_areas = []

    for c in cnts:
        area = cv2.contourArea(c)

        x, y, w, h = cv2.boundingRect(c)
        rect_area = w*h
        (x, y), radius = cv2.minEnclosingCircle(c)
        circ_area = np.pi * (radius ** 2)

        rect_diff = abs(area - rect_area)
        circ_diff = abs(area - circ_area)

        if(rect_diff > circ_diff):
            circle += 1
            countour_list.append(c)
            circ_areas += [circ_area]
        else:
            square += 1
            square_countour_list.append(c)
            rect_areas += [rect_area]

    print circle
    print square
    cntr_count = cntr_count
    avg_circle = sum(circ_areas) / float(len(circ_areas))
    avg_square = sum(rect_areas) / float(len(rect_areas))
    print "cntr count:" + str(cntr_count)
    print "circ_ares_Avg:" + str(avg_circle)
    print "square_areas_Avg:" + str(avg_square)
    # if square >= circle:
    #     print "square disease"
    # else:
    #     print "circular disease"
    #
    # cv2.drawContours(orig, square_countour_list, -1, (0, 255, 0), 1)
    # cv2.drawContours(orig, countour_list, -1, (255, 0, 0), 1)
    # cv2.imshow('objects detected', orig)
    # cv2.waitKey(0)

    return [filename, cntr_count, avg_circle, avg_square]
