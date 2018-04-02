# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
import operator
# from src.crowd_count import CrowdCounter
from src.crowd_count_mod_loss import CrowdCounter

from src import network
from src.data_loader import TestImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

# data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
# gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

data_path = './data/view116/all116/'
gt_path = './data/view116/all116_gt/'
# branch pre-train
model_path = './final_models/mcnn_shanghaiB_minibatch_42_ms.h5'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_all98_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name+"_all98")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0

# load test data
# downsample = True
data_loader = TestImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)

# downsample = False
# data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)
error={}
avgt_count=0
count=0
for blob in data_loader:                        
    im_data = blob['data']
    gt_data = blob['gt_density']
    density_map = net(im_data, gt_data,False)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)

    #Calculating average number of people
    avgt_count+=gt_count
    count=count+1
    new_dict={blob['fname']:abs(gt_count-et_count)/gt_count}
    error.update(new_dict)

    mae += abs(gt_count - et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))
print('\nAverage Number of people: %0.2f' %(avgt_count/count))
sort_error=sorted(error.items(),key=operator.itemgetter(1))

f = open(file_results, 'w')
f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))

for i in range(-20,0):
    print(sort_error[i])
    f.write(str(sort_error[i]))

f.close()
