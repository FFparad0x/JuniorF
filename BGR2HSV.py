import cv2
import numpy as np
color = np.uint8([[[17,59,129 ]]])
hsv_color = cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
print (hsv_color)