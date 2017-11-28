import cv2
import numpy as np

frame = cv2.imread("bals.png")
frame = np.uint8(frame)

def CopyRight(x,y):
    cv2.putText(res,(str(x) + " "+ str(y)),(x,y+30),2, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #Копировать правильно ©
def nothing(x):
    pass
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
cv2.createTrackbar('area', 'frame', 1, 500000, nothing)
while(1):

    # Take each frame
    MIN_area = cv2.getTrackbarPos('area', 'frame')
    h1 = cv2.getTrackbarPos('H1', 'frame')
    s1 = cv2.getTrackbarPos('S1', 'frame')
    v1 = cv2.getTrackbarPos('V1', 'frame')
    v2 = cv2.getTrackbarPos('H2', 'frame')
    s2 = cv2.getTrackbarPos('S2', 'frame')
    h2 = cv2.getTrackbarPos('V2', 'frame')
    a = cv2.getTrackbarPos('a','frame')
    b = cv2.getTrackbarPos("b","frame")
    d = cv2.getTrackbarPos('d', 'frame')
    c = cv2.getTrackbarPos("c", "frame")
    if a == 0:
        a =1
    b = cv2.getTrackbarPos("b", "frame")
    if b ==0:
        b = 1
    d = cv2.getTrackbarPos('d', 'frame')
    if d ==0:
        d = 1
    c = cv2.getTrackbarPos("c", "frame")
    if c ==0:
        c = 1

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([h1, s1, v1])
    upper_blue = np.array([h2, s2 , v2])

    #lghtYelow = np.uint8([[[215, 255, 255]]])

    #lellow = cv2.cvtColor(lghtYelow, cv2.COLOR_BGR2HSV)


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv,lower_blue,upper_blue)


    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(a,b)), iterations = 1)
    mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(c,d)),iterations= 1)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    ret, thresh = cv2.threshold(frame, 127, 255, 0)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refArea = 0
    index = 0
    numOjects = len(_)
    print(cv2.getWindowProperty('res',cv2.WND_PROP_ASPECT_RATIO))

    for index in range(numOjects):
        cnt = contours[index]
        M = cv2.moments(cnt)
        area = M['m00']
        if area> refArea:
            cx = int(M['m10'] / area)
            cy = int(M['m01'] / area)
            refArea = area
            cv2.drawContours(res, contours, -1, (0, 255, 0), 3)
            cv2.putText(res,"yeey",(0,50),2,1,(255,255,255),2,cv2.LINE_AA)
            CopyRight(cx,cy)
    # Bitwise-AND mask and original image


    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.resizeWindow('res',1200,1200)
    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()

