import cv2
import numpy as np

frame = cv2.imread("dead.jpg")
frame = np.uint8(frame)


def CopyRight(x, y):
    cv2.putText(res, (str(x) + " " + str(y)), (x, y + 30), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Копировать правильно ©


def nothing(x):
    pass


cv2.namedWindow("frame")
cv2.namedWindow("res")
cv2.createTrackbar('H1', 'res', 0, 255, nothing)
cv2.createTrackbar('S1', 'res', 60, 255, nothing)
cv2.createTrackbar('V1', 'res', 98, 255, nothing)
cv2.createTrackbar('H2', 'res', 255, 255, nothing)
cv2.createTrackbar('S2', 'res', 255, 255, nothing)
cv2.createTrackbar('V2', 'res', 255, 255, nothing)
cv2.createTrackbar('a', 'res', 1, 255, nothing)
cv2.createTrackbar('b', 'res', 1, 255, nothing)
cv2.createTrackbar('d', 'res', 1, 255, nothing)
cv2.createTrackbar('c', 'res', 1, 255, nothing)
cv2.createTrackbar('marea', 'res', 1, 100000, nothing)
cv2.createTrackbar('mxarea', 'res', 1, 1000000, nothing)
while (1):
    # Take each res
    MIN_area = cv2.getTrackbarPos('marea', 'res')
    MAx_area = cv2.getTrackbarPos('mxarea', 'res')
    h1 = cv2.getTrackbarPos('H1', 'res')
    s1 = cv2.getTrackbarPos('S1', 'res')
    v1 = cv2.getTrackbarPos('V1', 'res')
    v2 = cv2.getTrackbarPos('H2', 'res')
    s2 = cv2.getTrackbarPos('S2', 'res')
    h2 = cv2.getTrackbarPos('V2', 'res')
    a = cv2.getTrackbarPos('a', 'res')
    b = cv2.getTrackbarPos("b", "res")
    d = cv2.getTrackbarPos('d', 'res')
    c = cv2.getTrackbarPos("c", "res")
    if a == 0:
        a = 1
    b = cv2.getTrackbarPos("b", "res")
    if b == 0:
        b = 1
    d = cv2.getTrackbarPos('d', 'res')
    if d == 0:
        d = 1
    c = cv2.getTrackbarPos("c", "res")
    if c == 0:
        c = 1

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([h1, s1, v1])
    upper_blue = np.array([h2, s2, v2])

    # lghtYelow = np.uint8([[[215, 255, 255]]])

    # lellow = cv2.cvtColor(lghtYelow, cv2.COLOR_BGR2HSV)


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a, b)), iterations=1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, d)), iterations=1)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    ret, thresh = cv2.threshold(frame, 127, 255, 0)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    refArea = 0
    index = 0
    numOjects = len(contours)
    # print(len(_), len(contours))
    # print(cv2.getWindowProperty('res',cv2.WND_PROP_ASPECT_RATIO))

    for index in range(numOjects):
        cnt = contours[index]
        M = cv2.moments(cnt)
        area = M['m00']
        if area > MIN_area and area < MAx_area:
            cx = int(M['m10'] / area)
            cy = int(M['m01'] / area)
            refArea = area
            if area > MIN_area and area < MAx_area:
                cx = int(M['m10'] / area)
                cy = int(M['m01'] / area)
                refArea = area
                # cv2.drawContours(res, contours[index], -1, (0, 255, 0), 3)
                cv2.putText(frame, "object detected", (0, 50), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
                CopyRight(cx, cy)
                x = cv2.minAreaRect(cnt)
                print(int(x[1][0]), int(x[1][1]))
                box = cv2.boxPoints(x)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
                # pWidth = w
                # cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
                obj = 1



    cv2.imshow('frame', frame)

    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    #cv2.resizeWindow('res', 1200, 1200)
    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()
