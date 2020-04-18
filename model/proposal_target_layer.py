
import numpy as np
import numpy.random as npr
from easydict import EasyDict as edict

from lib.utils import bbox_transform
from lib.cuda.nms_wrapper import nms

# configurations
cfg = edict()
cfg.TRAIN = edict()
cfg.TEST = edict()
cfg.TEST.BBOX_NUM = 200
cfg.TEST.REGION_NMS_THRES = 0.5
cfg.TRAIN.BATCH_SIZE = 256
cfg.TRAIN.FG_FRACTION = 0.5
cfg.TRAIN.BATCH_SIZE_RELATIONSHIP = 512
cfg.TRAIN.FG_FRACTION_RELATIONSHIP = 0.5
cfg.TRAIN.REGION_NMS_THRES =0.5
cfg.TRAIN.FG_THRESH = 0.5
cfg.TRAIN.BG_THRESH_HI = 0.4
cfg.TRAIN.BG_THRESH_LO = 0.0

####

# from ..utils.cython_bbox import bbox_overlaps # later 


def merge_gt_rois(object_rois, gt_rois, thresh=0.5):
    overlaps = bbox_overlaps(
        np.ascontiguousarray(object_rois, dtype=np.float),
        np.ascontiguousarray(gt_rois, dtype=np.float))
    max_overlaps = overlaps.max(axis=1)
    keep_inds = np.where(max_overlaps < thresh)[0]
    rois = np.concatenate((gt_rois, object_rois[keep_inds]), 0)
    rois = rois[:len(object_rois)]
    return rois


def graph_construction(object_rois, gt_rois=None): 
    object_roi_num = min(cfg.TEST.BBOX_NUM, object_rois.shape[0])
    object_rois = object_rois[:object_roi_num]

    if gt_rois is not None:
        object_rois = merge_gt_rois(object_rois, gt_rois) # to make the message passing more likely to training
        sub_assignment, obj_assignment, _ = _generate_pairs(range(len(gt_rois)))
    else:
        sub_assignment=None
        obj_assignment=None

    object_rois, region_rois, mat_object, mat_relationship, mat_region = _setup_connection(object_rois,
            nms_thres=cfg.TEST.REGION_NMS_THRES,
            sub_assignment_select=sub_assignment,
            obj_assignment_select=obj_assignment)

    return object_rois, region_rois, mat_object, mat_relationship, mat_region


def proposal_target_layer(object_rois, gt_objects, gt_relationships, num_classes):
    num_images = 1

    object_rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    object_keep_inds, object_gt_assignment, object_fg_indicator, object_fg_duplicate = \
            _sample_rois(object_rois[:, 1:5], gt_objects[:, :4], object_rois_per_image, cfg.TRAIN.FG_FRACTION)

    object_labels = gt_objects[object_gt_assignment, 4]
    object_labels[np.logical_not(object_fg_indicator)] = 0
    object_selected_rois = object_rois[object_keep_inds]

    object_bbox_targets_temp = bbox_transform(object_selected_rois[:, 1:5], gt_objects[object_gt_assignment, :4])
    object_bbox_target_data = np.hstack(
        (object_labels[:, np.newaxis], object_bbox_targets_temp)).astype(np.float32, copy=False)
    object_bbox_targets, object_bbox_inside_weights = \
        _get_bbox_regression_labels(object_bbox_target_data, num_classes)

    rel_per_image = int(cfg.TRAIN.BATCH_SIZE_RELATIONSHIP / num_images)
    rel_bg_num = rel_per_image
    object_fg_inds = object_keep_inds[object_fg_indicator]
    if object_fg_inds.size > 0:
        id_i, id_j = np.meshgrid(xrange(object_fg_inds.size), xrange(object_fg_inds.size), indexing='ij') # Grouping the input object rois
        id_i = id_i.reshape(-1)
        id_j = id_j.reshape(-1)
        pair_labels = gt_relationships[object_gt_assignment[id_i], object_gt_assignment[id_j]]
        fg_id_rel = np.where(pair_labels > 0)[0]
        rel_fg_num = fg_id_rel.size
        rel_fg_num = int(min(np.round(rel_per_image * cfg.TRAIN.FG_FRACTION_RELATIONSHIP), rel_fg_num))

        if rel_fg_num > 0:
            fg_id_rel = npr.choice(fg_id_rel, size=rel_fg_num, replace=False)
        else:
            fg_id_rel = np.empty(0, dtype=int)

        rel_labels_fg = pair_labels[fg_id_rel]
        sub_assignment_fg = id_i[fg_id_rel]
        obj_assignment_fg = id_j[fg_id_rel]
        rel_bg_num = rel_per_image - rel_fg_num

    phrase_labels = np.zeros(rel_bg_num, dtype=np.float)
    sub_assignment = npr.choice(xrange(object_keep_inds.size), size=rel_bg_num, replace=True)
    obj_assignment = npr.choice(xrange(object_keep_inds.size), size=rel_bg_num, replace=True)
    if (sub_assignment == obj_assignment).any(): # an ugly hack for the issue
        obj_assignment[sub_assignment == obj_assignment] = (obj_assignment[sub_assignment == obj_assignment] + 1) % object_keep_inds.size

    if object_fg_inds.size > 0:
        phrase_labels = np.append(rel_labels_fg, phrase_labels, )
        sub_assignment = np.append(sub_assignment_fg, sub_assignment,)
        obj_assignment = np.append(obj_assignment_fg, obj_assignment, )

    object_selected_rois, region_selected_rois, mat_object, mat_relationship, mat_region = \
            _setup_connection(object_selected_rois,  nms_thres=cfg.TRAIN.REGION_NMS_THRES,
                                                sub_assignment_select = sub_assignment,
                                                obj_assignment_select = obj_assignment)

    object_labels = object_labels.reshape(-1, 1)
    phrase_labels = phrase_labels.reshape(-1, 1)
    object_fg_duplicate = np.stack([object_fg_indicator, object_fg_duplicate], axis=1)

    return (object_labels, object_selected_rois, object_bbox_targets, object_bbox_inside_weights, mat_object, object_fg_duplicate), \
              (phrase_labels, mat_relationship), \
              (region_selected_rois[:, :5], mat_region) \


def _sample_rois(rois, gt_rois, rois_per_image, fg_frac):
    assert rois.shape[1] == 4, 'Shape mis-match: [{}, {}] v.s. [:, 4]'.format(rois.shape[0], rois.shape[1])
    fg_rois_per_image = np.round(fg_frac * rois_per_image)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(rois, dtype=np.float),
        np.ascontiguousarray(gt_rois, dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)

    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    
    if bg_inds.size == 0:
        bg_inds = np.where(max_overlaps < cfg.TRAIN.BG_THRESH_HI)[0]

    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))

    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)

    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    if len(fg_inds) > 0:
        fg_overlap = max_overlaps[fg_inds]
        fg_sorted = np.argsort(fg_overlap)[::-1]
        fg_inds = fg_inds[fg_sorted]

    keep_inds = np.append(fg_inds, bg_inds)
    gt_assignment = gt_assignment[keep_inds]
    fg_indicator = np.zeros(len(keep_inds), dtype=np.bool)
    fg_indicator[:len(fg_inds)] = True

    _, highest_inds = np.unique(gt_assignment[:len(fg_inds)], return_index=True)
    fg_duplicate = np.zeros(len(keep_inds))
    fg_duplicate[highest_inds] = 1

    return keep_inds, gt_assignment, fg_indicator, fg_duplicate


def _setup_connection(object_rois,  nms_thres=0.6, sub_assignment_select = None, obj_assignment_select = None):
    sub_assignment, obj_assignment, rel_assignment = _generate_pairs(range(object_rois.shape[0]), sub_assignment_select, obj_assignment_select)
    region_rois = box_union(object_rois[sub_assignment], object_rois[obj_assignment])
    mapping = nms(region_rois[:, 1:].astype(np.float32), nms_thres, retain_all=True)

    keep, keep_inverse = np.unique(mapping, return_inverse=True)
    selected_region_rois = region_rois[keep, :5]

    mat_region = np.zeros((len(keep), object_rois.shape[0]), dtype=np.int64)
    mat_relationship = np.zeros((len(rel_assignment), 3), dtype=np.int64)
    mat_relationship[:, 0] = sub_assignment[rel_assignment]
    mat_relationship[:, 1] = obj_assignment[rel_assignment]
    mat_relationship[:, 2] = keep_inverse[rel_assignment]

    for relationship_id, region_id in enumerate(keep_inverse):
        mat_region[region_id, sub_assignment[relationship_id]] +=1
        mat_region[region_id, obj_assignment[relationship_id]] +=1

    mat_region = mat_region.astype(np.bool, copy=False)
    mat_object = mat_region.transpose()

    return object_rois[:, :5], selected_region_rois, mat_object, mat_relationship, mat_region


def box_union(box1, box2):
    return np.concatenate((
                np.minimum(box1[:, :3], box2[:, :3]),
                np.maximum(box1[:, 3:5], box2[:, 3:5]),
                box1[:, [5]] * box2[:, [5]]), 1)


def _generate_pairs(ids, sub_assignment_select = None, obj_assignment_select = None):
    id_i, id_j = np.meshgrid(ids, ids, indexing='ij')
    id_i = id_i.reshape(-1)
    id_j = id_j.reshape(-1)

    if sub_assignment_select is not None and obj_assignment_select is not None:
        rel_assignment = sub_assignment_select * len(ids) + obj_assignment_select
    else:
        rel_assignment = range(len(id_i))

    return id_i, id_j, rel_assignment


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = [1.0, 1.0, 1.0, 1.0]
    
    return bbox_targets, bbox_inside_weights
