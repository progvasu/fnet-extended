import os
import cPickle
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from lib.network import set_trainable_param
from lib.bbox_transform import bbox_transform_inv, clip_boxes

# configuration - used in rpn also - move to a common configuration file
from easydict import EasyDict as edict
cfg = edict()
cfg.TRAIN = edict()
cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
cfg.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
cfg.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

def save_results(results, epoch, dir_logs, is_testing = True):
    if is_testing:
        subfolder_name = 'evaluate_nms'
    else:
        subfolder_name = 'epoch_' + str(epoch)

    dir_epoch = os.path.join(dir_logs, subfolder_name)
    path_rslt = os.path.join(dir_epoch, 'testing_result.pkl')
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'wb') as f:
        cPickle.dump(results, f, cPickle.HIGHEST_PROTOCOL)


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





def unary_nms(dets, classes, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where((ovr <= thresh) | (classes[i] != classes[order[1:]]))[0]
        order = order[inds + 1]

    return keep

def triplet_nms_py(sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, thresh):
    #print('before: {}'.format(len(sub_ids))),
    sub_x1 = sub_boxes[:, 0]
    sub_y1 = sub_boxes[:, 1]
    sub_x2 = sub_boxes[:, 2]
    sub_y2 = sub_boxes[:, 3]
    obj_x1 = obj_boxes[:, 0]
    obj_y1 = obj_boxes[:, 1]
    obj_x2 = obj_boxes[:, 2]
    obj_y2 = obj_boxes[:, 3]


    sub_areas = (sub_x2 - sub_x1 + 1) * (sub_y2 - sub_y1 + 1)
    obj_areas = (obj_x2 - obj_x1 + 1) * (obj_y2 - obj_y1 + 1)
    order = np.array(range(len(sub_ids)))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        sub_xx1 = np.maximum(sub_x1[i], sub_x1[order[1:]])
        sub_yy1 = np.maximum(sub_y1[i], sub_y1[order[1:]])
        sub_xx2 = np.minimum(sub_x2[i], sub_x2[order[1:]])
        sub_yy2 = np.minimum(sub_y2[i], sub_y2[order[1:]])
        sub_id = sub_ids[i]
        obj_xx1 = np.maximum(obj_x1[i], obj_x1[order[1:]])
        obj_yy1 = np.maximum(obj_y1[i], obj_y1[order[1:]])
        obj_xx2 = np.minimum(obj_x2[i], obj_x2[order[1:]])
        obj_yy2 = np.minimum(obj_y2[i], obj_y2[order[1:]])
        obj_id = obj_ids[i]
        pred_id = pred_ids[i]

        w = np.maximum(0.0, sub_xx2 - sub_xx1 + 1)
        h = np.maximum(0.0, sub_yy2 - sub_yy1 + 1)
        inter = w * h
        sub_ovr = inter / (sub_areas[i] + sub_areas[order[1:]] - inter)

        w = np.maximum(0.0, obj_xx2 - obj_xx1 + 1)
        h = np.maximum(0.0, obj_yy2 - obj_yy1 + 1)
        inter = w * h
        obj_ovr = inter / (obj_areas[i] + obj_areas[order[1:]] - inter)
        inds = np.where( (sub_ovr <= thresh) |
                                    (obj_ovr <= thresh) |
                                    (sub_ids[order[1:]] != sub_id) |
                                    (obj_ids[order[1:]] != obj_id) |
                                    (pred_ids[order[1:]] != pred_id) )[0]
        order = order[inds + 1]
    #print(' After: {}'.format(len(keep)))
    return sub_ids[keep], obj_ids[keep], pred_ids[keep], sub_boxes[keep], obj_boxes[keep], keep


def nms_detections(pred_boxes, scores, nms_thresh, inds):

    dets = np.hstack((pred_boxes, scores[:, np.newaxis])).astype(np.float32)
    keep = unary_nms(dets, inds, nms_thresh)
    # print('NMS: [{}] --> [{}]'.format(scores.shape[0], len(keep)))
    keep = keep[:min(100, len(keep))]
    return pred_boxes[keep], scores[keep], inds[keep], keep

def interpret_relationships(cls_prob, bbox_pred, rois, cls_prob_predicate,
                        	mat_phrase, im_info, nms=-1., clip=True, min_score=0.01,
                        	top_N=100, use_gt_boxes=False, triplet_nms=-1., topk=10, 
                            reranked_score=None):

        scores, inds = cls_prob[:, 1:].data.max(1)
        if reranked_score is not None:
            if isinstance(reranked_score, Variable):
                reranked_score = reranked_score.data
            scores *= reranked_score.squeeze()
        inds += 1
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        predicate_scores, predicate_inds = cls_prob_predicate[:, 1:].data.topk(dim=1, k=topk)
        predicate_inds += 1
        predicate_scores, predicate_inds = predicate_scores.cpu().numpy().reshape(-1), predicate_inds.cpu().numpy().reshape(-1)


        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        keep = range(scores.shape[0])
        if use_gt_boxes:
            triplet_nms = -1.
            pred_boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
        else:
            pred_boxes = bbox_transform_inv(rois.data.cpu().numpy()[:, 1:5], box_deltas) / im_info[0][2]
            pred_boxes = clip_boxes(pred_boxes, im_info[0][:2] / im_info[0][2])

            # nms
            if nms > 0. and pred_boxes.shape[0] > 0:
                assert nms < 1., 'Wrong nms parameters'
                pred_boxes, scores, inds, keep = nms_detections(pred_boxes, scores, nms, inds=inds)


        sub_list = np.array([], dtype=int)
        obj_list = np.array([], dtype=int)
        pred_list = np.array([], dtype=int)

        # mapping the object id
        mapping = np.ones(cls_prob.size(0), dtype=np.int64) * -1
        mapping[keep] = range(len(keep))


        sub_list = mapping[mat_phrase[:, 0]]
        obj_list = mapping[mat_phrase[:, 1]]
        pred_remain = np.logical_and(sub_list >= 0,  obj_list >= 0)
        pred_list = np.where(pred_remain)[0]
        sub_list = sub_list[pred_remain]
        obj_list = obj_list[pred_remain]

        # expand the sub/obj and pred list to k-column
        pred_list = np.vstack([pred_list * topk + i for i in range(topk)]).transpose().reshape(-1)
        sub_list = np.vstack([sub_list for i in range(topk)]).transpose().reshape(-1)
        obj_list = np.vstack([obj_list for i in range(topk)]).transpose().reshape(-1)

        if use_gt_boxes:
            total_scores = predicate_scores[pred_list]
        else:
            total_scores = predicate_scores[pred_list] * scores[sub_list] * scores[obj_list]

        top_N_list = total_scores.argsort()[::-1][:10000]
        total_scores = total_scores[top_N_list]
        pred_ids = predicate_inds[pred_list[top_N_list]] # category of predicates
        sub_assignment = sub_list[top_N_list] # subjects assignments
        obj_assignment = obj_list[top_N_list] # objects assignments
        sub_ids = inds[sub_assignment] # category of subjects
        obj_ids = inds[obj_assignment] # category of objects
        sub_boxes = pred_boxes[sub_assignment] # boxes of subjects
        obj_boxes = pred_boxes[obj_assignment] # boxes of objects


        if triplet_nms > 0.:
            sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, keep = triplet_nms_py(sub_ids, obj_ids, pred_ids, sub_boxes, obj_boxes, triplet_nms)
            sub_assignment = sub_assignment[keep]
            obj_assignment = obj_assignment[keep]
            total_scores = total_scores[keep]
        if len(sub_list) == 0:
            print('No Relatinoship remains')
            #pdb.set_trace()

        return pred_boxes, scores, inds, sub_ids, obj_ids, sub_boxes, obj_boxes, pred_ids, sub_assignment, obj_assignment, total_scores
