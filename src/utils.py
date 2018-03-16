# -*- coding:utf-8 -*-

import cv2
import numpy as np
import os
import torch

def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):
    # input_img = input_img[0][0]
    # print('input size = ', input_img.shape)

    gt_data = 255*gt_data/np.max(gt_data)

    # transfer to 3 channels
    gt_data = torch.from_numpy(gt_data)
    gt_data.unsqueeze_(2)
    gt_data = gt_data.repeat(1,1,3)
    gt_data = gt_data.numpy()
    # print('gt_data size = ', gt_data.shape)

    density_map = 255*density_map/np.max(density_map)

    # gt_data = gt_data[0][0]

    density_map = density_map[0][0]

    # transfer to 3 channels
    density_map = torch.from_numpy(density_map)
    density_map.unsqueeze_(2)
    density_map = density_map.repeat(1,1,3)
    density_map = density_map.numpy()
    density_map = density_map.astype(np.float32, copy=False)
    # print('density_map size = ', density_map.shape)

    # print('density_map.shape[1]',density_map.shape[1],'\n')
    # print('gt_data.shape[1]', gt_data.shape[1], '\n')
    # print('input_img.shape[1]', input_img.shape[1], '\n')

#    if density_map.shape[1] == input_img.shape[1]:
#        return

    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)
    

def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)

def save_demo_density_map(density_map,output_dir, fname='results.png'):
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    density_map=density_map.astype(np.uint8)
    density_map=cv2.applyColorMap(density_map,cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    # input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    # gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
