import cv2
import numpy as np

cap = cv2.VideoCapture(0)
def CopyRight(x,y):
    cv2.putText(res,(str(x) + " "+ str(y)),(x,y+30),2, 1, (255, 255, 255), 2, cv2.LINE_AA)

def nothing(x):
    pass
cv2.namedWindow("trcks")
cv2.namedWindow("tracks2")
cv2.resizeWindow('trcks', 400, 400)
cv2.resizeWindow('traks2', 400, 400)
cv2.createTrackbar('H1', 'trcks', 0, 255, nothing)
cv2.createTrackbar('S1', 'trcks', 174, 255, nothing)
cv2.createTrackbar('V1', 'trcks', 71, 255, nothing)
cv2.createTrackbar('H2', 'trcks', 255, 255, nothing)
cv2.createTrackbar('S2', 'trcks', 233, 255, nothing)
cv2.createTrackbar('V2', 'trcks', 17, 255, nothing)
cv2.createTrackbar('a', 'tracks2', 3, 255, nothing)
cv2.createTrackbar('b', 'tracks2', 4, 255, nothing)
cv2.createTrackbar('d', 'tracks2', 8, 255, nothing)
cv2.createTrackbar('c', 'tracks2', 5, 255, nothing)
cv2.createTrackbar('contours','tracks2',0,1,nothing)
cv2.createTrackbar('MIN_area', 'tracks2', 1, 1000, nothing)
cv2.createTrackbar('MAX_area', 'tracks2', 1, 2000, nothing)
while(1):

    # Take each frame
    _, frame = cap.read()
    MIN_area = cv2.getTrackbarPos('area', 'tracks2')
    MAX_area = cv2.getTrackbarPos('area', 'tracks2')
    h1 = cv2.getTrackbarPos('H1', 'trcks')
    s1 = cv2.getTrackbarPos('S1', 'trcks')
    v1 = cv2.getTrackbarPos('V1', 'trcks')
    v2 = cv2.getTrackbarPos('H2', 'trcks')
    s2 = cv2.getTrackbarPos('S2', 'trcks')
    h2 = cv2.getTrackbarPos('V2', 'trcks')
    a = cv2.getTrackbarPos('a', 'tracks2')
    if a == 0:
        a =1
    b = cv2.getTrackbarPos("b", "tracks2")
    if b ==0:
        b = 1
    d = cv2.getTrackbarPos('d', 'tracks2')
    if d ==0:
        d = 1
    c = cv2.getTrackbarPos("c", "tracks2")
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
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a, b)), iterations=1)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c, d)), iterations=1)


   # kernel1 = np.ones((a, b), np.uint8)
    #kernel2 = np.ones((c, d), np.uint8)
   # maks = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    #maks = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    # Bitwise-AND mask and original image



    '''ghf'''
    res = cv2.bitwise_and(frame,frame, mask= mask)
    if cv2.getTrackbarPos('contours','tracks2') == 1:
        ret, thresh = cv2.threshold(frame, 127, 255, 0)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refArea = 0
        index = 0
        numOjects = len(_)
        print(cv2.getWindowProperty('res', cv2.WND_PROP_ASPECT_RATIO))

        for index in range(numOjects):
            cnt = contours[index]
            M = cv2.moments(cnt)
            area = M['m00']
            if area > refArea and area > MIN_area and area < MAX_area:
                cx = int(M['m10'] / area)
                cy = int(M['m01'] / area)
                refArea = area
                cv2.drawContours(frame, contours[index], -1, (0, 255, 0), 3)
                cv2.putText(frame, "yeey", (0, 50), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)
                CopyRight(cx, cy)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()