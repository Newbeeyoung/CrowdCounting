import torch.nn as nn
from src import network

from src.model import MCNN

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()        
        self.DME = MCNN()

        self.alpha = 0.0001
        self.loss_mse = nn.MSELoss(size_average=True)
        self.loss_L1 = nn.L1Loss(size_average=True)
        # self.loss_fn = 0.5 * nn.MSELoss() + self.alpha * nn.L1Loss()

    @property
    def loss(self):
        return self.loss_all
    
    def forward(self,  im_data, gt_data,is_training):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_density=False, is_training=is_training)
        density_map = self.DME(im_data)
        
        if is_training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_density=True,is_training=is_training)
            self.loss_all = self.build_loss(density_map, gt_data)
            # self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        # loss = self.loss_fn(density_map, gt_data)
        loss_mse = 0.5 * self.loss_mse(density_map, gt_data)
        loss_L1 = self.alpha * self.loss_L1(density_map, gt_data)
        return loss_mse + loss_L1
