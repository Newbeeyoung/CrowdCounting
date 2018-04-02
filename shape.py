import cv2
import torch
import pandas as pd
import os

#Test shape of image

# path='./data/shanghaiB/outdoor_test/images/'
# path1='./data/original/shanghaitech/part_A_final/test_data/images/IMG_2.jpg'
# for file in os.listdir(path):
#     ori=cv2.imread(path+file)
#     print(ori.shape)
# den = pd.read_csv(path, sep=',',header=None).as_matrix()
# den = den.astype(np.float32, copy=False)
# den=den*255
# print(den.shape)
# ori=cv2.imread(path1)
# print("original:"+str(ori.shape))
# cv2.imshow("original",ori)
# cv2.imshow("density",den)
# cv2.waitKey(0)

#rename
# path='./data/formatted_trainval/mall_dataset/rgb_train/'
#
# for fname in os.scandir(path):
#     f=str(int(fname.path[6:10]))
#     f=f+'.jpg'
#     print(f)
    # finalname=os.path.join(path,f)
    # print(finalname)
    # os.rename(fname,finalname)
# import tkinter as tk
# import os
#
# path=os.path.abspath(tk.__file__)
# print(path)

import numpy as np

a=np.array([1,2])
b=np.array([2,3])
c=np.array([3,4])

d=[]
d.append(a)
d.append(b)
d.append(c)

print(np.array(d))