from pickletools import uint1
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms


def genTrimap(mask,k_size=(5,5),ite=1):
    # 膨張収縮処理
    kernel = np.ones(k_size,np.uint8)
    eroded = cv2.erode(mask,kernel,iterations = ite)
    dilated = cv2.dilate(mask,kernel,iterations = ite)
    # 膨張収縮の差はグレーに着色
    trimap = np.full(mask.shape,128)
    trimap[eroded == 255] = 255
    trimap[dilated == 0] = 0
    return trimap


#関数化
def trimapFunction(img):
    #BGR->RGBへ変換
    img = img[...,::-1]
    h,w,_ = img.shape
    #画像のリサイズ
    img = cv2.resize(img,(320,320))
    #GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()

    #学習済みモデルの読み込み
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
    ])

    inputTensor = preprocess(img)
    inputBatch = inputTensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(inputBatch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    mask[mask!=0] = 255
    img = cv2.resize(img,(w,h))
    plt.gray()
    plt.subplot(1,2,1)
    #plt.imshow(img)
    plt.subplot(1,2,2)
    #plt.imshow(mask);

    plt.figure(figsize=(20,20))
    
    plt.subplot(1,2,1)
    #plt.imshow(img)
    plt.subplot(1,2,2)
    #plt.imshow(trimap)
    #cv2.imwrite('do_capture_folder/mask.png', mask)
    return mask    

