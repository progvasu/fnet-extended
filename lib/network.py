import pdb

import numpy as np

import torch
import torch.nn as nn

def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
                print ('[Copied]: {}').format(k)
            else:
                print ('[Missed]: {}').format(k)
                print ('[Manually copy instructions]: \n'
                         'check the existence of new name:\n'
                         '\t \'{}\' in h5f\n'
                         'if True, then copy\n'
                         '\t param = torch.from_numpy(np.asarray(h5f[\'{}\']))\n'
                         '\t v.copy_(param)\n'.format(k, k))
                pdb.set_trace()
        except Exception as e:
            print (e)
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...').format(k)
            pdb.set_trace()

            
class GroupDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False, group=16):
        super(GroupDropout, self).__init__()
        self.group = group
        # cannot use inplace operation: [.view] is used
        self.inplace = False
        self.p = p

    def forward(self, x):
        if self.training and self.p > 1e-5:
            assert x.size(1) % self.group == 0, "Channels should be divided by group number [{}]".format(self.group)
            original_size = x.size()
            if x.dim() == 2:
                x = x.view(x.size(0), self.group, x.size(1) / self.group, 1)
            else:
                x = x.view(x.size(0), self.group,  x.size(1) / self.group * x.size(2), x.size(3))
            x = nn.functional.dropout2d(x, p=self.p, inplace=self.inplace, training=True)
            x = x.view(original_size)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
        
def set_trainable(model, requires_grad):
    set_trainable_param(model.parameters(), requires_grad)
def set_trainable_param(parameters, requires_grad):
    for param in parameters:
        param.requires_grad = requires_grad


def weight_init_fun_kaiming(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in')
        m.bias.data.fill_(0.)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.)
def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    return np_to_tensor(x, is_cuda, dtype)
def np_to_tensor(x, is_cuda=True, dtype=torch.FloatTensor):
    v = torch.from_numpy(x).type(dtype)
    if is_cuda:
        v = v.cuda()
    return v


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg
class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum