# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
import sys
import random
from src.crowd_count_mod_loss import CrowdCounter

from src import network
from src.data_loader import ImageDataLoader,TestImageDataLoader
from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



method = 'mcnn'

dataset_name = 'shanghaiB_minibatch'
output_dir = './saved_models/'

train_result=os.path.join(output_dir,dataset_name+'_train_minibatch.txt')
branch = 'ms'



# gray training images
# train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train'
# train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/train_den'
# train_path = './data/formatted_trainval/shanghaitech_part_A_patches_100/train'
# train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_100/train_den'

# rgb training images
# train_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/rgb_train'
# train_gt_path = './data/formatted_trainval/shanghaitech_part_A_patches_9/rgb_train_den'
train_path = './data/shanghaiB/outdoor_train/rgb_train_9'
train_gt_path = './data/shanghaiB/outdoor_train/rgb_train_den_9'

test_path = './data/shanghaiB/outdoor_test/images'
test_gt_path = './data/shanghaiB/outdoor_test/ground_truth'

#  training configuration
start_step = 0
end_step = 2000

# lr = 0.00001
lr = 0.00001
BATCH_SIZE=16

momentum = 0.9
disp_interval = 100
# disp_interval = 5000
log_interval = 250


#  Tensorboard  config
use_tensorboard = True
save_exp_name = method + '_' + dataset_name + '_' + 'v1'
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------
rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
#    torch.cuda.manual_seed(rand_seed)


# load net
net = CrowdCounter()

# network parameter init
#network.weights_normal_init(net, dev=0.01)

# load pretained model
model_path = './final_models/mcnn_shanghaiB_2_ms.h5'

# loading pre-train model
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)

net.cuda()
net.train()

params = list(net.parameters())

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:    
        exp_name = save_exp_name 
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

# downsample = True
data_loader = ImageDataLoader(train_path, train_gt_path,batch_size=BATCH_SIZE, shuffle=True, gt_downsample=True, pre_load=True)
# data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
data_loader_test = TestImageDataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True, pre_load=True)

# downsample = False
# data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=False, pre_load=True)
# data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True, pre_load=True)
# data_loader_test = ImageDataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=False, pre_load=True)


best_mae = sys.maxsize

f=open(train_result,'w+')
for epoch in range(start_step, end_step+1):
# for epoch in range(1):
    step = -1
    train_loss = 0
    for blob in data_loader:                
        step = step + 1        
        im_data = np.array(blob['data'])
        gt_data = np.array(blob['gt_density'])
        density_map = net(im_data, gt_data,True)

        loss = net.loss
        train_loss += loss.data[0]
        step_cnt += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration
            step_cnt = 0
            gt_count = np.sum(gt_data)
            density_map = density_map.data.cpu().numpy()
            et_count = np.sum(density_map)
            # utils.save_results(im_data, gt_data, density_map, output_dir)
            log_text = 'epoch: %4d, step %4d, Time: %.4fs, gt_cnt: %4.1f, et_cnt: %4.1f' % (epoch,
                step, 1./fps, gt_count, et_count)
            log_print(log_text, color='green', attrs=['bold'])
            re_cnt = True


        if re_cnt:
            t.tic()
            re_cnt = False


    if (epoch % 2 == 0):
        save_name = os.path.join(output_dir, '{}_{}_{}_{}.h5'.format(method,dataset_name,epoch,branch))
        network.save_net(save_name, net)
        #  calculate error on the validation dataset
        mae,mse = evaluate_model(save_name, data_loader_test)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}_{}.h5'.format(method,dataset_name,epoch,branch)
        log_text = 'EPOCH: %d, MAE: %.2f, MSE: %.2f' % (epoch,mae,mse)

        f.write(log_text)
        log_print(log_text, color='green', attrs=['bold'])
        log_text = 'BEST MAE: %.2f, BEST MSE: %.2f, BEST MODEL: %s' % (best_mae,best_mse, best_model)
        log_print(log_text, color='green', attrs=['bold'])
        if use_tensorboard:
            exp.add_scalar_value('MAE', mae, step=epoch)
            exp.add_scalar_value('MSE', mse, step=epoch)
            exp.add_scalar_value('train_loss', train_loss/data_loader.get_num_samples(), step=epoch)
f.close()