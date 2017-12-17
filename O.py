import cv2
import numpy as np


cap = cv2.VideoCapture(0)

def CopyRight(x, y):
    cv2.putText(res, (str(x) + " " + str(y)), (x, y + 30), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Копировать правильно ©


def nothing(x):
    pass


print("Ширина и Высота ")
rWidth, rHeight = float(input()), float(input())
print("Фокусное расстояние")
fLength = float(input())

cv2.namedWindow("frame")
cv2.createTrackbar('H1', 'frame', 0, 255, nothing)
cv2.createTrackbar('S1', 'frame', 60, 255, nothing)
cv2.createTrackbar('V1', 'frame', 98, 255, nothing)
cv2.createTrackbar('H2', 'frame', 255, 255, nothing)
cv2.createTrackbar('S2', 'frame', 255, 255, nothing)
cv2.createTrackbar('V2', 'frame', 255, 255, nothing)
cv2.createTrackbar('a', 'frame', 1, 255, nothing)
cv2.createTrackbar('b', 'frame', 1, 255, nothing)
cv2.createTrackbar('d', 'frame', 1, 255, nothing)
cv2.createTrackbar('c', 'frame', 1, 255, nothing)
cv2.createTrackbar('marea', 'frame', 1, 100000, nothing)
cv2.createTrackbar('mxarea', 'frame', 1, 1000000, nothing)
io, cFrame = cap.read()

while (1):
    _, frame = cap.read()
    # Take each frame
    frame = np.uint8(frame)
    MIN_area = cv2.getTrackbarPos('marea', 'frame')
    MAx_area = cv2.getTrackbarPos('mxarea', 'frame')
    h1 = cv2.getTrackbarPos('H1', 'frame')
    s1 = cv2.getTrackbarPos('S1', 'frame')
    v1 = cv2.getTrackbarPos('V1', 'frame')
    v2 = cv2.getTrackbarPos('H2', 'frame')
    s2 = cv2.getTrackbarPos('S2', 'frame')
    h2 = cv2.getTrackbarPos('V2', 'frame')
    a = cv2.getTrackbarPos('a', 'frame')
    b = cv2.getTrackbarPos("b", "frame")
    d = cv2.getTrackbarPos('d', 'frame')
    c = cv2.getTrackbarPos("c", "frame")
    if a == 0:
        a = 1
    b = cv2.getTrackbarPos("b", "frame")
    if b == 0:
        b = 1
    d = cv2.getTrackbarPos('d', 'frame')
    if d == 0:
        d = 1
    c = cv2.getTrackbarPos("c", "frame")
    if c == 0:
        c = 1

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([h1, s1, v1])
    upper_blue = np.array([h2, s2, v2])



    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # mask= cv2.Canny(mask, 35, 125)

    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a, b)), iterations=1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, d)), iterations=1)

    res = cv2.bitwise_and(frame, frame, mask=mask)
    ret, thresh = cv2.threshold(frame, 127, 255, 0)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    refArea = 0
    index = 0
    numOjects = len(contours)
    pWidth = 0
    obj = 0
    # print(cv2.getWindowProperty('res',cv2.WND_PROP_ASPECT_RATIO))

    for index in range(numOjects):
        cnt = contours[index]
        M = cv2.moments(cnt)
        area = M['m00']
        if area > MIN_area and area < MAx_area:
            cx = int(M['m10'] / area)
            cy = int(M['m01'] / area)
            refArea = area
            #cv2.drawContours(res, contours[index], -1, (0, 255, 0), 3)
            cv2.putText(res, "yeey", (0, 50), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
            CopyRight(cx, cy)
            x = cv2.minAreaRect(cnt)
            print(int(x[1][0]), int(x[1][1]))
            box = cv2.boxPoints(x)
            box = np.int0(box)
            cv2.drawContours(res, [box], 0, (0, 0, 255), 2)
            # pWidth = w
            # cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 2)
            obj = 1
        else:
            obj = 0
    # Bitwise-AND mask and original image
    if obj == 1:
        pWidth = x[1][0]
        if x[1][0] <= x[1][1]:
            Distance = (rWidth * fLength) / pWidth
        else:
            Distance = (rHeight * fLength) / pWidth
        print(Distance)
    else:
        print("Не удалось найти объект")
    # print(Distance)
    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()

cap.release()
