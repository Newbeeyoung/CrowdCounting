# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
import cv2
# from src.crowd_count import CrowdCounter
from src.crowd_count_mod_loss import CrowdCounter

from src import network
from src.data_loader import SingleImageDataLoader
from src import utils
from src.timer import Timer

t=Timer()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

# data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
# gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

def single_img_estimate(input_path):
    data_path = input_path
    gt_path = './data/formatted_trainval/mall_dataset/rgb_val_den/'+input_path.split('/')[-1].replace('.jpg','.csv')
    # branch pre-train
    model_path = './final_models/mcnn_mall_perspective_28_ms.h5'

    output_dir = './demo_output/'
    gt_dir='./demo_gt/'
    model_name = os.path.basename(model_path).split('.')[0]
    file_results = os.path.join(output_dir, 'results_' + model_name + '_.txt')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)
    gt_dir = os.path.join(gt_dir, 'density_maps_' + model_name)
    if not os.path.exists(gt_dir):
        os.mkdir(gt_dir)

    net = CrowdCounter()

    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    # load test data
    # downsample = True
    data_loader = SingleImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)

    # downsample = False
    # data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        t.tic()
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        duration=t.toc()
        print ("time duration:"+str(duration))
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_demo_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

            gt_data = 255 * gt_data / np.max(gt_data)
            gt_data= gt_data.astype(np.uint8)
            gt_data = cv2.applyColorMap(gt_data,cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(gt_dir,'gt_'+blob['fname'].split('.')[0]+'.png'),gt_data)

        print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
        f = open(file_results, 'w')
        f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
        f.close()

        return (output_dir + '/output_' + blob['fname'].split('.')[0] + '.png',
                gt_dir+'/gt_'+blob['fname'].split('.')[0]+'.png',mae,mse,gt_count,et_count)
