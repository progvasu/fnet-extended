import torch
import time
import yaml
import cPickle as pickle

from lib import network
from model.rpn import RPN
from lib.metrics import check_recall
from lib.network import np_to_variable

from datasets.VRD import VRD
import argparse
import lib.utils_rpn as RPN_utils

import pdb

parser = argparse.ArgumentParser('Options for training RPN in pytorch')

## training settings
parser.add_argument('--path_data_opts', type=str, default='options/data_VRD.yaml', help='Dataset scale and max_size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=15, metavar='N', help='max iterations for training')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='')
parser.add_argument('--log_interval', type=int, default=500, help='Interval for Logging')
parser.add_argument('--disable_clip_gradient', action='store_true', help='Whether to clip the gradient')
# parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors') # similar param 'kmeans_anchors' present in config file
parser.add_argument('--step_size', type=int, default=5, help='step to decay the learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='#images per batch') # should be 1 otherwise RPN fails
parser.add_argument('--workers', type=int, default=4)
## environment settings
parser.add_argument('--output_dir', type=str, default='./output/RPN', help='Location to output the model')
parser.add_argument('--model_name', type=str, default='RPN_VRD', help='model name for snapshot')
parser.add_argument('--resume', type=str, help='The model we resume')
parser.add_argument('--path_rpn_opts', type=str, default='options/RPN_FN_VRD.yaml', help='Path to RPN opts')
parser.add_argument('--evaluate', action='store_true', help='To enable the evaluate mode')
args = parser.parse_args()

def main():
    global args

    print ("Loading training set and testing set...")
    with open(args.path_data_opts, 'r') as f:
        data_opts = yaml.load(f, Loader=yaml.FullLoader)
    train_set = VRD(data_opts, 'train', batch_size=args.batch_size)
    test_set = VRD(data_opts, 'test', batch_size=args.batch_size)
    print ("done")

    # getting RPN architectural settings
    with open(args.path_rpn_opts, 'r') as f:
        opts = yaml.load(f, Loader=yaml.FullLoader)
        opts['scale'] = train_set.opts['test']['SCALES'][0]
        print('scale: {}'.format(opts['scale']))

    # create network architecture
    net = RPN(opts)

    # pass enough message for anchor target generation???
    train_set._feat_stride = net._feat_stride
    train_set._rpn_opts = net.opts

    train_loader = torch.utils.data.DataLoader(train_set, 
                                                batch_size=args.batch_size,
                                                shuffle=False if args.evaluate else True, 
                                                num_workers=args.workers,
                                                pin_memory= True, 
                                                collate_fn=VRD.collate)
    test_loader = torch.utils.data.DataLoader(test_set, 
                                                batch_size=args.batch_size,
                                                shuffle=False, 
                                                num_workers=args.workers,
                                                pin_memory=True, 
                                                collate_fn=VRD.collate)

    if args.resume is not None:
        print('Resume training from: {}'.format(args.resume))
        RPN_utils.load_checkpoint(args.resume, net)
        optimizer = torch.optim.SGD([{'params': list(net.parameters())[26:]},], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    else:
        print ('Training from scratch...')
        # optimize just the last layers... what about the last?
        optimizer = torch.optim.SGD(list(net.parameters())[26:], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    # if all features were set not be trained from scratch
    # then why fix first 4 layers separately in VRD data loader? old code present there?
    network.set_trainable(net.features, requires_grad=False)
    net.cuda()

    if args.evaluate:
        net.eval()
        # test (train_loader, net)
        test (test_loader, net)
        return


def test(test_loader, target_net):
    box_num = 0
    correct_cnt, total_cnt = 0., 0.
    print ('========== Testing ==========')

    results = []

    batch_time = network.AverageMeter()
    end = time.time()
    im_counter = 0
    for i, sample in enumerate(test_loader):
        correct_cnt_t, total_cnt_t = 0., 0.

        im_data = sample['visual'][0].cuda()
        im_counter += im_data.size(0)
        im_info = sample['image_info']
        gt_objects = sample['objects']

        object_rois = target_net(im_data, im_info, gt_objects=gt_objects, image_name=sample['path'][0])[1]

        results.append(object_rois.cpu().data.numpy())

        box_num += object_rois.size(0)
        correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_objects, 200)
        correct_cnt += correct_cnt_t
        total_cnt += total_cnt_t
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 100 == 0 and i > 0:
            print('[{0}/{6}]  Time: {1:2.3f}s/img).'
                  '\t[object] Avg: {2:2.2f} Boxes/im, Top-200 recall: {3:2.3f} ({4:.0f}/{5:.0f})'.format(
                    i + 1, batch_time.avg,
                    box_num / float(im_counter), correct_cnt / float(total_cnt)* 100, correct_cnt, total_cnt,
                    len(test_loader)))

    recall = correct_cnt / float(total_cnt)
    print ('====== Done Testing ======')
    return recall, results

if __name__ == '__main__':
    main()