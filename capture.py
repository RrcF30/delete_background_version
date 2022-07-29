import cv2
from pickletools import uint1
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

import torch
import torchvision
from torchvision import transforms

import alpha
import trimap


capture = cv2.VideoCapture(0)

def captureCamera():
    ret, frame = capture.read()

    if ret == False:
        print("some error occured.")
        exit(1)
    return frame

while(True):
    frame = captureCamera()

    #cv2.imwrite('do_capture_folder/frame.png',frame)
    #print(frame[0])
    

    mask = trimap.trimapFunction(frame)

    cv2.imwrite("do_capture_folder/mask.png",mask)
    mask =cv2.imread("do_capture_folder/mask.png")

    alpha.alphaFunction(frame,mask)


    result = cv2.imread("do_capture_folder/delete.png")

    print(type(result))
    print(result.shape)

    cv2.imshow("output", result)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)
    

capture.release()
cv2.destroyAllWindows()