import json
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from lib.utils_rpn import generate_output_mapping, build_loss, proposal_layer as proposal_layer_py, reshape_layer
from lib import network
from lib.network import Conv2d

class RPN(nn.Module):
    _feat_stride = 16

    anchor_scales_normal = [2, 4, 8, 16, 32, 64]
    anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]

    def __init__(self, opts):
        super(RPN, self).__init__()
        self.opts = opts

        # only kmeans anchors are used - nothing else!
        kmeans_anchors_file = osp.join(self.opts['anchor_dir'], 'kmeans_anchors.json')
        print ('using k-means anchors: {}'.format(kmeans_anchors_file))
        anchors = json.load(open(kmeans_anchors_file))
            
        if 'scale' not in self.opts:
            print('No RPN scale is given, default [600] is set')
            
        self.opts['object']['anchor_scales'] = list(np.array(anchors['anchor_scales_kmeans']) / 600.0 * self.opts.get('scale', 600.))
        self.opts['object']['anchor_ratios'] = anchors['anchor_ratios_kmeans']

        self.anchor_num = len(self.opts['object']['anchor_scales'])

        self.features = models.vgg16(pretrained=True).features
        self.features.__delattr__('30') # deleting last max pooling
        network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False)

        # rpn specific layers
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, self.anchor_num * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, self.anchor_num * 4, 1, relu=False, same_padding=False)

        self.initialize_parameters()

        self.opts['mappings'] = generate_output_mapping(osp.join(self.opts['anchor_dir'], 'vgg16_mappings.json'), self.features)

    def initialize_parameters(self, normal_method='normal'):
        if normal_method == 'normal':
            normal_fun = network.weights_normal_init
        else:
            raise(Exception('Initialization method not implemented: {}'.format(normal_method)))

        normal_fun(self.conv1, 0.025)
        normal_fun(self.score_conv, 0.025)
        normal_fun(self.bbox_conv, 0.01)

    def forward(self, im_data, im_info, gt_objects=None, dontcare_areas=None, rpn_data=None, image_name=None):
        features = self.features(im_data) # feature-map
        rpn_conv1 = self.conv1(features)
        
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob_reshape = reshape_layer(rpn_cls_prob, self.anchor_num*2)

        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        cfg_key = 'train' if self.training else 'test'
        rois = self.proposal_layer(rpn_cls_prob_reshape, 
                                   rpn_bbox_pred, 
                                   im_info,
                                   self._feat_stride, 
                                   self.opts['object'][cfg_key],
                                   self.opts['object']['anchor_scales'],
                                   self.opts['object']['anchor_ratios'],
                                   mappings=self.opts['mappings'],
                                   image_name=image_name,
                                   gt_objects=gt_objects)

        losses = {}
        if self.training and rpn_data is not None:
            self.loss_cls, self.loss_box, accs = build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)
            self.tp, self.tf, self.fg_cnt, self.bg_cnt = accs
            
            losses = {
                'loss_cls': self.loss_cls,
                'loss_box': self.loss_box,
                'loss': self.loss_cls + self.loss_box * 0.2,
                'tp': self.tp,
                'tf': self.tf,
                'fg_cnt': self.fg_cnt,
                'bg_cnt': self.bg_cnt
            }

        return features, rois, losses

    @property
    def loss(self):
        return self.loss_cls + self.loss_box * 0.2

    @staticmethod
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, _feat_stride, opts, anchor_scales, anchor_ratios, mappings, image_name=None, gt_objects=None):
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()
        
        x = proposal_layer_py(rpn_cls_prob_reshape, 
                                rpn_bbox_pred, 
                                im_info, 
                                _feat_stride, 
                                opts, 
                                anchor_scales, 
                                anchor_ratios, 
                                mappings, 
                                image_name=image_name, 
                                gt_objects=gt_objects)
        
        x = network.np_to_variable(x, is_cuda=True)
        return x.view(-1, 6)
