import cv2
import torch
import numpy as np

import pandas as pd
import os

#Test shape of image

# path='./data/formatted_trainval/mall_dataset/rgb_val_den/52.csv'
# path1='./data/formatted_trainval/mall_dataset/rgb_val/52.jpg'
# den = pd.read_csv(path, sep=',',header=None).as_matrix()
# den = den.astype(np.float32, copy=False)
# den=den*255
# print(den.shape)
# ori=cv2.imread(path1)
# print("original:"+str(ori.shape))
# cv2.imshow("original",ori)
# img=cv2.imread(path)
# cv2.imshow("image",path)
# # img=np.transpose(img,(2,0,1))
# print(img.shape)
# v=torch.from_numpy(img)
# v.unsqueeze_(0)
# print(v.shape)
# cv2.imshow("density",den)
# cv2.waitKey(0)

#rename
path='./data/formatted_trainval/mall_dataset/rgb_train/'

for fname in os.scandir(path):
    f=str(int(fname.path[6:10]))
    f=f+'.jpg'
    print(f)
    # finalname=os.path.join(path,f)
    # print(finalname)
    # os.rename(fname,finalname)
