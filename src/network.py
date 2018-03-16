import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Deconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, relu=True, same_padding=False, bn=False):
        super(Deconv2D, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#TODO:
class rbLSTM(nn.LSTM):
    def __init__(self):
        pass

    def forward(self):
        pass


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)

def np_to_variable(x, is_cuda=True, is_training=False, is_density=False, dtype=torch.FloatTensor):
    ## Orginal One
    v = torch.from_numpy(x)

    # gray image
    # v.unsqueeze_(0)
    # v.unsqueeze_(0)
    if is_density:
        v.unsqueeze_(0)
        v.unsqueeze_(0)
    else:
        if len(v.shape) == 3:
            if v.shape[2] == 3:
                # rgb image
                v.unsqueeze_(0)
                v = v.permute(0, 3, 1, 2)
    # elif len(v.shape) == 2:
    #     v.unsqueeze_(0)
    #     v.unsqueeze_(0)
    # # print(v.type)



    # gray image
    # v.unsqueeze_(0)
    # v.unsqueeze_(0)
    #
    # if len(x.shape) == 3:
    #     if x.shape[2] == 3:
    #         v = torch.from_numpy(x)
    #         # rgb image
    #         v.unsqueeze_(0)
    #         v = v.permute(0, 3, 1, 2)
    # elif len(x.shape) == 2:
    #     x=np.stack((x,)*3,-1)
    #     x=np.transpose(x,(2,0,1))
    #     v=torch.from_numpy(x)
    #     v.unsqueeze_(0)
    # print(v.type)


    if is_training:
        v = Variable(v.type(dtype))
    else:
        v = Variable(v.type(dtype), requires_grad = False, volatile = True)

    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
