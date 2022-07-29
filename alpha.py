import numpy as np
import cv2
import matplotlib.pyplot as plt

def alphaFunction(img,mask):
    #img = img[...,::-1]
    #mask = cv2.imread("do_capture_folder/mask.png")
    h,w,_ = img.shape
    bg = np.full_like(img,255)
    
    #print(mask.shape)
    #print(img.shape)

        
    img = img.astype(float)
    bg = bg.astype(float)

    mask = mask.astype(float)/255
    img = cv2.multiply(img, mask)
    bg = cv2.multiply(bg, 1.0 - mask)
    outImage = cv2.add(img, bg)
    cv2.imwrite("do_capture_folder/delete.png",outImage)