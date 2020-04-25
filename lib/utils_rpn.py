import pdb
import json
import os.path as osp

import numpy as np
import numpy.random as npr

import torch
import torch.nn.functional as F

from lib.bbox_transform import bbox_transform, bbox_transform_inv, clip_boxes
from lib.cuda.nms_wrapper import nms

from lib.cuda.cython_bbox import bbox_overlaps

def load_checkpoint(filename, model):
    model_name = '{}.h5'.format(filename)
    load_net_from_fnet(model_name, model)

def load_net_from_fnet(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        try:
            if ('rpn.'+k) in h5f:
                param = torch.from_numpy(np.asarray(h5f['rpn.'+k]))
                v.copy_(param)
                print ('[Copied]: {}').format(k)
            else:
                print ('[Missed]: {}').format(k)
                # print ('[Manually copy instructions]: \n'
                #          'check the existence of new name:\n'
                #          '\t \'{}\' in h5f\n'
                #          'if True, then copy\n'
                #          '\t param = torch.from_numpy(np.asarray(h5f[\'{}\']))\n'
                #          '\t v.copy_(param)\n'.format(k, k))
                # pdb.set_trace()
        except Exception as e:
            print (e)
            print ('[Loaded net not complete] Parameter[{}] Size Mismatch...').format(k)
            pdb.set_trace()


def reshape_layer(x, d):
    input_shape = x.size()
    x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
    )
    return x


def build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
    # classification loss
    rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
    rpn_label = rpn_data[0].view(-1)

    rpn_keep = rpn_label.data.ne(-1).nonzero().squeeze()
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

    fg_cnt = torch.sum(rpn_label.data.ne(0))
    bg_cnt = rpn_label.data.numel() - fg_cnt

    _, predict = torch.max(rpn_cls_score.data, 1)

    if fg_cnt == 0:
        tp = 0.
        tf = torch.sum(predict.eq(rpn_label.data))
    else:
        tp = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
        tf = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
    
    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

    # box loss
    rpn_bbox_targets, rpn_bbox_inside_weights, _ = rpn_data[1:]
    rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
    rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
    rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, reduction='sum') /  (fg_cnt + 1e-4)

    return rpn_cross_entropy, rpn_loss_box, (tp, tf, fg_cnt, bg_cnt)

def generate_output_mapping(mapping_file, conv_layers, min_size=16, max_size=1001):
    """
        Get mappings from image dimension (height to width) to the resultant dimension i.e. the model output
    """
    if osp.isfile(mapping_file):
        with open(mapping_file, 'r') as f:
            mappings = json.load(f)
        mappings = {int(k):int(v) for k,v in mappings.items()}
    else:
        print ("need to generate - refer org. code")
    
    return mappings

#### functions specific to proposal_layer

def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_infos, _feat_stride, opts, anchor_scales, anchor_ratios, mappings,
    image_name=None, gt_objects=None):
    """
        Generate bbox's for RPN, also applies NMS
    """
    batch_size = rpn_cls_prob_reshape.shape[0]
    _anchors = generate_anchors(scales=anchor_scales, ratios=anchor_ratios)
    _num_anchors = _anchors.shape[0]
    pre_nms_topN = opts['num_box_pre_NMS']
    post_nms_topN = opts['num_box_post_NMS']
    nms_thres = opts['nms_thres']
    min_size = opts['min_size']

    blob = []
    
    for i in range(batch_size):
        im_info = im_infos[i]
        height = mappings[int(im_info[0])]
        width = mappings[int(im_info[1])]
        
        scores = rpn_cls_prob_reshape[i, _num_anchors:, :height, :width]
        bbox_deltas = rpn_bbox_pred[i, :, :height, :width]

        shift_x = np.arange(0, width) * _feat_stride
        shift_y = np.arange(0, height) * _feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        A = _num_anchors
        K = shifts.shape[0] # number of possible shifts
        anchors = _anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4)) # represents all possible gt anchor boxes at all locations

        bbox_deltas = bbox_deltas.transpose((1, 2, 0)).reshape((-1, 4))
        scores = scores.transpose((1, 2, 0)).reshape((-1, 1))
        proposals = bbox_transform_inv(anchors, bbox_deltas)

        if opts['dropout_box_runoff_image']:
            _allowed_border = 16
            inds_inside = np.where(
                (proposals[:, 0] >= -_allowed_border) &
                (proposals[:, 1] >= -_allowed_border) &
                (proposals[:, 2] < im_info[1] + _allowed_border) &  # width
                (proposals[:, 3] < im_info[0] + _allowed_border)  # height
            )[0]
            proposals = proposals[inds_inside, :]
        
        proposals = clip_boxes(proposals, im_info[:2])

        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        order = scores.ravel().argsort()[::-1]
        # if pre_nms_topN > 0:
        #     order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        nms_thres = 0.6
        keep = nms(np.hstack((proposals, scores)).astype(np.float32), nms_thres)
        # if post_nms_topN > 0:
        #     keep = keep[:post_nms_topN]
            # keep = keep[:10] # trying to reduce memory requirement

        proposals = proposals[keep, :]
        scores = scores[keep]

        # USING NO12000 RPN-INDI_TRAINED SCORES
        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/data/vrd/test-no12000-nms-0.7/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:100])
        # scores = np.array(jsonC['scores'])
        # proposals = proposals[:100]
        # scores = scores[:100]

        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:20]]
        # scores = scores[sorted_indexs[:20]].reshape(20, 1)

        # storing the correct post-nms boxes
        # saving post-nms and thresholding
        if image_name:
            pre_nms_store = {}
            pre_nms_store['gt_boxes'] = gt_objects[0].tolist()
            pre_nms_store['detections'] = proposals.tolist()
            pre_nms_store['scores'] = scores.tolist()
            pre_nms_store['scale'] = im_info[2]
            print ("no detections: {}, saving...".format(len(proposals.tolist())))
            with open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/data/vrd/test-fnet-no12000-nms0.6/" + image_name.split('.')[0] + ".json", 'w') as f:
                json.dump(pre_nms_store, f)  

        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/output-fnet-0.6-no-350-0.3-4-full/test_vrd/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:350])
        # scores = np.array(jsonC['predictions'])
        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:10]]
        # scores = scores[sorted_indexs[:10]].reshape(10, 1)

        # mini = scores.min()
        # maxi = scores.max()
        # scores = (scores - mini) / (maxi - mini)
        # scores = np.ones([100, 1])

        # the entire code will go away and based on file_name we need to feed in detections

        # config 0. - top 350
        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/output-0.7-0.5-350-0.3-4-full/test_vrd/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:350])
        # scores = np.array(jsonC['predictions'])
        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:10]]
        # scores = scores[sorted_indexs[:10]].reshape(10, 1)
        # mini = scores.min()
        # maxi = scores.max()
        # scores = (scores - mini) / (maxi - mini)
        # scores = np.ones([100, 1])


        # config 1. - top 2000 
        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/output-0.8-2000(50)-0.3-4-full/test_vrd/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:2000])
        # scores = np.array(jsonC['predictions'])
        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:20]]
        # scores = scores[sorted_indexs[:20]].reshape(20, 1)


        # config 2. - cc 
        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/output-0.8-2000(cc)-0.2-4-full/test_vrd/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:400])
        # scores = np.array(jsonC['predictions'])
        # # print (np.array(jsonC['detections']).shape)
        # # print (scores.shape)
        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:20]]
        # scores = scores[sorted_indexs[:20]].reshape(20, 1)

        # config 3. - cn 
        # jsonC = json.load(open("/home/vasu/Desktop/Thesis/gossipnet-pytorch/output-0.8-2000(cn)-0.3-4-full/test_vrd/" + image_name + ".json"))
        # proposals = np.array(jsonC['detections'][:2000])
        # scores = np.array(jsonC['predictions'])
        # sorted_indexs = np.argsort(-scores)
        # proposals = proposals[sorted_indexs[:20]]
        # scores = scores[sorted_indexs[:20]].reshape(20, 1)


        batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * i

        blob.append(np.hstack((batch_inds, proposals.astype(np.float32, copy=False), scores.astype(np.float32, copy=False))))

    return np.concatenate(blob, axis=0)

def _filter_boxes(boxes, min_size):
    """
        Remove all boxes with any side smaller than min_size
    """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    
    return keep

def generate_anchors(ratios, scales, base_size=16):
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    w, h, x_ctr, y_ctr = _whctrs(base_anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    ws = ws * np.array(scales) # print (ws.shape) (25, 1)
    hs = hs * np.array(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    ws = ws[:, np.newaxis] # print (ws.shape) # (25, 1)
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1), x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
    return anchors





#### anchor target layer


def anchor_target_layer(img, gt_boxes, im_info, _feat_stride, rpn_opts, mappings):
    _anchors = generate_anchors(scales=rpn_opts['anchor_scales'], ratios=rpn_opts['anchor_ratios'])
    opts = rpn_opts['train']
    _num_anchors = _anchors.shape[0]
    full_height, full_width = mappings[int(img.size(1))], mappings[int(img.size(2))]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = opts['allowed_border']

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, full_width) * _feat_stride
    shift_y = np.arange(0, full_height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    height = mappings[int(im_info[0])]
    width = mappings[int(im_info[1])]
    valid_mask =np.zeros((full_height, full_width, A), dtype=np.bool)
    valid_mask[:height, :width] = True
    valid_ids = valid_mask.reshape(K*A)
    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border) &  # height
        valid_ids # remove the useless points
    )[0]
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(np.logical_and(overlaps == gt_max_overlaps, overlaps > 0.))[0] # avoid zero overlap

    if not opts['clobber_positives']:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < opts['negative_overlap']] = 0

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    labels[max_overlaps >= opts['positive_overlap']] = 1
    if opts['clobber_positives']:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < opts['negative_overlap']] = 0

    # subsample positive labels if we have too many
    num_fg = int(opts['fg_fraction'] * opts['batch_size'])
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
    else:
        num_fg = len(fg_inds)
        
    # subsample negative labels if we have too many
    num_bg = opts['batch_size'] - num_fg
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) < num_bg:
        disable_inds = npr.choice(
            disable_inds, size=(len(disable_inds) - num_bg + len(bg_inds)), replace=False)
    try:
        labels[disable_inds] = -1
    except Exception:
        pass

    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(opts['BBOX_INSIDE_WEIGHTS'])

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if opts['POSITIVE_WEIGHT'] < 0:
        num_examples = np.sum(labels >= 0)
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((opts['POSITIVE_WEIGHT'] > 0) &
                (opts['POSITIVE_WEIGHT'] < 1))
        positive_weights = (opts['POSITIVE_WEIGHT'] /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - opts['POSITIVE_WEIGHT']) /
                            np.sum(labels == 0))
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    labels = labels.reshape((full_height, full_width, A)).transpose(2, 0, 1).reshape(-1)
    bbox_targets = bbox_targets.reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)

    bbox_inside_weights = bbox_inside_weights.reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)
    bbox_outside_weights = bbox_outside_weights.reshape((full_height, full_width, A * 4)).transpose(2, 0, 1)

    result = [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights]
    return result


def _unmap(data, count, inds, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] >= 4

    targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

    return targets