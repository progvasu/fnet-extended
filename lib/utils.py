import os
import numpy as np

import torch
import torch.nn.functional as F

from lib.network import set_trainable_param

# configuration - used in rpn also - move to a common configuration file
from easydict import EasyDict as edict
cfg = edict()
cfg.TRAIN = edict()
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)


def update_values(dict_from, dict_to):
	for key, value in dict_from.items():
		if isinstance(value, dict):
			update_values(dict_from[key], dict_to[key])
		elif value is not None:
			dict_to[key] = dict_from[key]

	return dict_to


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def group_features(net_):
    # fixed features of vgg16 in rpn network
    vgg_features_fix = list(net_.rpn.features.parameters())[:8]
    
    # variable features of vgg16 in - trained along with FN
    vgg_features_var = list(net_.rpn.features.parameters())[8:]
        
    vgg_feature_len = len(list(net_.rpn.features.parameters()))
    rpn_feature_len = len(list(net_.rpn.parameters())) - vgg_feature_len 
    rpn_features = list(net_.rpn.parameters())[vgg_feature_len:]

    fn_features = list(net_.parameters())[(rpn_feature_len + vgg_feature_len):]
    mps_features = list(net_.mps_list.parameters())
    fn_features = list(set(fn_features) - set(mps_features))

    return vgg_features_fix, vgg_features_var, rpn_features, fn_features, mps_features


def get_optimizer(lr, mode, opts, vgg_features_var, rpn_features, fn_features, mps_features=[]):
    """ 
            mode: 2 - fixed - resume training
    """
    fn_features += mps_features

    set_trainable_param(vgg_features_var, True)
    set_trainable_param(rpn_features, True)
    set_trainable_param(fn_features, True)
        
    if opts['optim']['optimizer'] == 0:
        optimizer = torch.optim.SGD([
                        {'params': rpn_features},
                        {'params': vgg_features_var, 'lr': lr * 0.1},
                        {'params': fn_features},
                    ], lr=lr, momentum=opts['optim']['momentum'], weight_decay=0.0005, nesterov=opts['optim']['nesterov'])
    elif opts['optim']['optimizer'] == 1:
        optimizer = torch.optim.Adam([
            {'params': rpn_features},
            {'params': vgg_features_var, 'lr': lr * 0.1},
            {'params': fn_features},
            ], lr=lr, weight_decay=0.0005)
    elif opts['optim']['optimizer'] == 2:
        optimizer = torch.optim.Adagrad([
            {'params': rpn_features},
            {'params': vgg_features_var, 'lr': lr * 0.1},
            {'params': fn_features},
            ], lr=lr, weight_decay=0.0005)
    else:
        raise Exception('Unrecognized optimization algorithm specified!')

    return optimizer


def get_model_name(opts):
    """
        Return model name - just adding few configuration names
    """
    model_name = opts['logs']['model_name']

    if opts['model'].get('use_kernel', False):
        model_name += '_with_kernel'

    model_name += '_SGD'

    opts['logs']['dir_logs'] = os.path.join(opts['logs']['dir_logs'], model_name)

    return opts


def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()


    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))

    return targets


def build_loss_cls(cls_score, labels, loss_weight=None):
        labels = labels.squeeze()
        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt
        cross_entropy = F.cross_entropy(cls_score, labels, weight=loss_weight)
        _, predict = cls_score.data.max(1)
        if fg_cnt == 0:
            tp = torch.zeros_like(fg_cnt)
        else:
            tp = torch.sum(predict[:fg_cnt].eq(labels.data[:fg_cnt]))
        tf = torch.sum(predict[fg_cnt:].eq(labels.data[fg_cnt:]))
        fg_cnt = fg_cnt
        bg_cnt = bg_cnt
        return cross_entropy, (tp, tf, fg_cnt, bg_cnt)

def build_loss_bbox(bbox_pred, roi_data, fg_cnt):
        bbox_targets, bbox_inside_weights = roi_data[2:4]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-5)
        return loss_box