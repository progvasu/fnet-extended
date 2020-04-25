import os
import os.path as osp
import random
import argparse
import yaml
import click
from pprint import pprint

import torch

import datasets
import model as models
from lib.utils import get_model_name, group_features, get_optimizer, save_results
import lib.network as network
import lib.utils as utils
from model.fnet import FactorizableNetwork
import lib.logger as logger

####

# import lib.utils.logger as logger
# from models.HDN_v2.utils import save_checkpoint, load_checkpoint, save_results, save_detections

parser = argparse.ArgumentParser('Options for training factorizable-net model in pytorch')

# model architecture and training parameter options
parser.add_argument('--path_opt', default='options/default_path', type=str, help='path to a yaml options file')

# directory to store the logs
parser.add_argument('--dir_logs', type=str, help='dir logs')
parser.add_argument('--model_name', type=str, help='model name prefix')
parser.add_argument('--dataset_option', type=str, help='data split selection [small | fat | normal]')
parser.add_argument('--workers', type=int, default=4, help='#dataloader workers')

# training parameters
parser.add_argument('-lr', '--learning_rate', type=float, help='initial learning rate')
parser.add_argument('--epochs', type=int, metavar='N', help='max iterations for training')
parser.add_argument('--eval_epochs', type=int, default= 1, help='Number of epochs to evaluate the model')
parser.add_argument('--print_freq', type=int, default=1000, help='Interval for Logging')
parser.add_argument('--step_size', type=int, help='Step size for decay learning rate')
parser.add_argument('--optimizer', type=int, choices=range(0, 3), help='Step size for decay learning rate')
parser.add_argument('-i', '--infinite', action='store_true', help='To enable infinite training')
parser.add_argument('--iter_size', type=int, default=1, help='Iteration size to update parameters')
parser.add_argument('--loss_weight', default=True)
parser.add_argument('--disable_loss_weight', dest='loss_weight', action='store_false', help='Set the dropout rate.')
parser.add_argument('--clip_gradient', default=True)
parser.add_argument('--disable_clip_gradient', dest='clip_gradient', action='store_false', help='Enable clip gradient')

# model parameters
parser.add_argument('--dropout', type=float, help='Set the dropout rate.')

# model init
parser.add_argument('--resume', type=str, help='path to latest checkpoint')
parser.add_argument('--pretrained_model', type=str, help='path to pretrained_model')
parser.add_argument('--warm_iters', type=int, default=-1, help='Indicate the model do not need')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('--use_normal_anchors', action='store_true', help='Whether to use kmeans anchors')

# environment settings
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--rpn', type=str, help='The Model used for initialize')
parser.add_argument('--nms', type=float, default=-1., help='NMS threshold for post object NMS (negative means not NMS)')
parser.add_argument('--triplet_nms', type=float, default=0.4, help='Triplet NMS threshold for post object NMS (negative means not NMS)')

# testing settings
parser.add_argument('--use_gt_boxes', action='store_true', help='Use ground truth bounding boxes for evaluation')

args = parser.parse_args()

is_best = False
best_recall = [0., 0.]
best_recall_phrase = [0., 0.]
best_recall_pred = [0., 0.]

def main():
    global args, is_best, best_recall, best_recall_pred, best_recall_phrase

    # set options
    options = {
        'logs': {
            'model_name': args.model_name,
            'dir_logs': args.dir_logs,
        },
        'data':{
            'dataset_option': args.dataset_option,
            'batch_size': torch.cuda.device_count(),
        },
        'optim': {
            'lr': args.learning_rate,
            'epochs': args.epochs,
            'lr_decay_epoch': args.step_size,
            'optimizer': args.optimizer,
            'clip_gradient': args.clip_gradient,
        },
        'model':{
            'dropout': args.dropout,
            'use_loss_weight': args.loss_weight,
        },
    }

    if args.path_opt is not None:
        # is the option file specified - if yes, read and update the parameters
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        options = utils.update_values(options, options_yaml)

        # read the dataset configuration file and add the data parameters i.e. scale and max_size
        with open(options['data']['opts'], 'r') as f:
            data_opts = yaml.load(f)
            options['opts'] = data_opts

    pprint (vars(args)) # cmd arguements
    pprint (options) # options from config file

    lr = options['optim']['lr']
    options = get_model_name(options) # get model name modified by network parameters
    print ('Checkpoints are saved to: {}'.format(options['logs']['dir_logs']))

    random.seed(args.seed)
    torch.manual_seed(args.seed + 1)
    torch.cuda.manual_seed(args.seed + 2)

    print ("Loading training set and testing set...")
    train_set = getattr(datasets, options['data']['dataset'])(data_opts, 'train')    
    test_set = getattr(datasets, options['data']['dataset'])(data_opts, 'test')

    # getattr(models, options_yaml['model']['arch']): FN_v4s for VRD
    model = getattr(models, options_yaml['model']['arch'])(train_set, opts=options['model'])

    train_set._feat_stride = model.rpn._feat_stride # 16 - standard RPN
    train_set._rpn_opts = model.rpn.opts

    # memory issue
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=options['data']['batch_size'],
                                                shuffle=True, num_workers=args.workers,
                                                pin_memory=True,
                                                collate_fn=getattr(datasets, options['data']['dataset']).collate, 
                                                drop_last=True,)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                                shuffle=False, num_workers=args.workers,
                                                pin_memory=True,
                                                collate_fn=getattr(datasets, options['data']['dataset']).collate)

    # print ("Training data: " + str(len(train_loader)))
    print ("Testing data: " + str(len(test_loader))) 

    _, vgg_features_var, rpn_features, fn_features, mps_features = group_features(model)
    network.set_trainable(model, False)
    exp_logger = None

    if args.resume is not None:
        # right now not going with this one
        print('Loading saved model: {}'.format(os.path.join(options['logs']['dir_logs'], args.resume)))
        args.train_all = True
        optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, fn_features, mps_features)
        args.start_epoch, best_recall[0], exp_logger = load_checkpoint(model, optimizer, os.path.join(options['logs']['dir_logs'], args.resume))
    else:
        # setting up the logs directory
        if os.path.isdir(options['logs']['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'.format(options['logs']['dir_logs'], default=False)):
                os.system('rm -r ' + options['logs']['dir_logs'])
            else:
                return
        os.system('mkdir -p ' + options['logs']['dir_logs'])
        
        # saving options and arguments files in log directory
        path_new_opt = os.path.join(options['logs']['dir_logs'], os.path.basename(args.path_opt))
        path_args = os.path.join(options['logs']['dir_logs'], 'args.yaml')
        
        with open(path_new_opt, 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

        # 3. if we have some initialization points
        if args.pretrained_model is not None:
            print('Loading pretrained model: {}'.format(args.pretrained_model))
            args.train_all = True
            network.load_net(args.pretrained_model, model)
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, fn_features, mps_features)
        
        # 4. training with pretrained RPN
        elif args.rpn is not None:
            print('Loading pretrained RPN: {}'.format(args.rpn))
            args.train_all = False
            network.load_net(args.rpn, model.rpn)
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, fn_features, mps_features)
        
        assert args.start_epoch == 0, 'Set [start_epoch] to 0, or something unexpected will happen.'

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=options['optim']['lr_decay_epoch'], gamma=options['optim']['lr_decay'])
    
    model.cuda()
    model.train()

    if exp_logger is None:
        exp_name = os.path.basename(options['logs']['dir_logs'])
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(exp_logger.info['model_params']))

    # recall 50 and recall 100
    top_Ns = [10, 20, 50, 100]

    if args.evaluate:
        model.eval()
        recall, result = model.engines.test(test_loader, 
                                                model, 
                                                top_Ns, 
                                                nms=args.nms, 
                                                triplet_nms=args.triplet_nms,
                                                use_gt_boxes=args.use_gt_boxes)
        
        print ('======= Testing Result =======, config-0, top-10, nms-rpn')
        for idx, top_N in enumerate(top_Ns):
            print ('Top-%d Recall \t[Pred]: %2.3f%% \t[Phr]: %2.3f%% \t[Rel]: %2.3f%%' % (
                        top_N, 
                        float(recall[2][idx]) * 100,
                        float(recall[1][idx]) * 100,
                        float(recall[0][idx]) * 100))
        print ('============ Done ============')
        
        # save_results(result, None, options['logs']['dir_logs'], is_testing=True)
        return

    print ('\n========== [Starting Training] ==========\n')

    for epoch in range(args.start_epoch, options['optim']['epochs']):
        # training
        scheduler.step()
        print('[Learning Rate]\t{}'.format(optimizer.param_groups[0]['lr']))
        is_best=False

        model.module.engines.train(
            train_loader, 
            model, 
            optimizer, 
            exp_logger, 
            epoch, 
            args.train_all, 
            args.print_freq,
            clip_gradient=options['optim']['clip_gradient'], iter_size=args.iter_size
        )

        if (epoch + 1) % args.eval_epochs == 0:
            print('\n============ Epoch {} ============'.format(epoch))
            recall, result = model.module.engines.test(test_loader, model, top_Ns,
                                                                nms=args.nms,
                                                                triplet_nms=args.triplet_nms)
            
            is_best = (recall[0] > best_recall).all()
            best_recall = recall[0] if is_best else best_recall
            best_recall_phrase = recall[1] if is_best else best_recall_phrase
            best_recall_pred = recall[2] if is_best else best_recall_pred
            print('\n[Result]')
            for idx, top_N in enumerate(top_Ns):
                print('\tTop-%d Recall'
                    '\t[Pred]: %2.3f%% (best: %2.3f%%)'
                    '\t[Phr]: %2.3f%% (best: %2.3f%%)'
                    '\t[Rel]: %2.3f%% (best: %2.3f%%)' % (
                        top_N, float(recall[2][idx]) * 100, float(best_recall_pred[idx]) * 100,
                        float(recall[1][idx]) * 100, float(best_recall_phrase[idx]) * 100,
                        float(recall[0][idx]) * 100, float(best_recall[idx]) * 100 ))

            save_checkpoint({
                    'epoch': epoch,
                    'arch': options['model']['arch'],
                    'exp_logger': exp_logger,
                    'best_recall': best_recall[0],
                },
                model.module, #model.module.state_dict(),
                optimizer.state_dict(),
                options['logs']['dir_logs'],
                args.save_all_from,
                is_best)
            print('====================================')


        # updating learning policy
        if (epoch + 1) == args.warm_iters:
            print('Free the base CNN part.')
            # options['optim']['clip_gradient'] = False
            args.train_all = True
            # update optimizer and correponding requires_grad state
            optimizer = get_optimizer(lr, 2, options, vgg_features_var, rpn_features, fn_features, mps_features)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                            step_size=options['optim']['lr_decay_epoch'],
                            gamma=options['optim']['lr_decay'])

def make_meters():
    meters_dict = {
        'loss': network.AverageMeter(),
        'loss_rpn': network.AverageMeter(),
        'loss_cls_obj': network.AverageMeter(),
        'loss_reg_obj': network.AverageMeter(),
        'loss_cls_rel': network.AverageMeter(),
        'loss_cls_cap': network.AverageMeter(),
        'loss_reg_cap': network.AverageMeter(),
        'loss_cls_objectiveness': network.AverageMeter(),
        'batch_time': network.AverageMeter(),
        'data_time': network.AverageMeter(),
        'epoch_time': network.SumMeter(),
        'best_recall': network.AverageMeter(),
    }
    return meters_dict

if __name__ == '__main__':
    main()
