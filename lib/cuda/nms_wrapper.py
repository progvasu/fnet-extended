import torch

from lib.cuda import _C
from lib.cuda.nms_retain_all import nms_retain_all

def nms(dets, thresh, retain_all=False):
    """
        Dispatch to GPU NMS implementations
    """
    if dets.shape[0] == 0:
        return []

    if retain_all:
    	return nms_retain_all(dets, thresh)
    else:
        dets = torch.Tensor(dets).cuda()
        return _C.nms(dets[:, :4], dets[:, 4], thresh).cpu().numpy()