# import required libraries
import serial
import struct
import random
import time
import re
import numpy as np

data = serial.Serial('com4', 9600, timeout=1)

for i in range(0, 180):
    pos = i
    # data.write(struct.pack('>B', 1))
    data.write(struct.pack('>B', pos))
    print(data.read_all())
    time.sleep(0.2)


    # data.write(struct.pack('>B',179)) #code and send the angle to the Arduino through serial port
