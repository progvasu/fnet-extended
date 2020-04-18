from PIL import Image
import os
import os.path as osp
import numpy as np
import numpy.random as npr
import json
import cv2

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
        
from lib.utils_rpn import anchor_target_layer

class VRD(data.Dataset):
    def __init__(self, opts, image_set='train', batch_size=1, dataset_option=None, use_region=False):
        '''
            batch_size: not used
            _rpn_opts: remove later - confirm across code
        '''
        super(VRD, self).__init__()
        self._name = image_set
        self.opts = opts
        self._image_set = image_set
        self._data_path = osp.join(self.opts['dir'], 'sg_{}_images'.format(image_set)) # opts['dir'] points to -> "data/images/vrd"
        
        annotation_dir = "/home/vasu/Desktop/Thesis/FactorizableNet/data/annotations/vrd"
        
        # load class inverse weights
        inverse_weight = json.load(open(osp.join(annotation_dir, 'inverse_weight.json')))
        self.inverse_weight_object = torch.FloatTensor(inverse_weight['object'])
        self.inverse_weight_predicate = torch.FloatTensor(inverse_weight['predicate'])
        
        # load annotation file -> train.json
        ann_file_path = osp.join(annotation_dir, self.name + '.json')
        self.annotations = json.load(open(ann_file_path))

        # load category information (class names + 'background_class')
        obj_cats = json.load(open(osp.join(annotation_dir, 'objects.json')))
        self._object_classes = tuple(['__background__'] + obj_cats)
        pred_cats = json.load(open(osp.join(annotation_dir, 'predicates.json')))
        self._predicate_classes = tuple(['__background__'] + pred_cats)
        # add indices to the class names
        self._object_class_to_ind = dict(zip(self.object_classes, xrange(self.num_object_classes)))
        self._predicate_class_to_ind = dict(zip(self.predicate_classes, xrange(self.num_predicate_classes)))

        # image transformation
        # how is this normalization calculated? - across training set
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize,])

        self.cfg_key = image_set.split('_')[0] # 'train' only
        self._feat_stride = None
        self._rpn_opts = None

    def __getitem__(self, index):
        '''
            Return a single image to the data loader, each image is randomly scaled
        '''
        item = {'rpn_targets': {}}
        item['path']= self.annotations[index]['path']

        # select a random scale value
        target_scale = self.opts[self.cfg_key]['SCALES'][npr.randint(0, high=len(self.opts[self.cfg_key]['SCALES']))]
        
        # read in the image
        img = cv2.imread(osp.join(self._data_path, item['path']))
        # print (osp.join(self._data_path, item['path']))

        img_original_shape = img.shape
        # print (img_original_shape)
        
        # resize the image according to the scale
        img, im_scale = self._image_resize(img, target_scale, self.opts[self.cfg_key]['MAX_SIZE'])
        
        # store the [image_height(new), image_width(new), scale_factor, image_height(original), image_width(original)]
        item['image_info'] = np.array([img.shape[0], img.shape[1], im_scale, img_original_shape[0], img_original_shape[1]], dtype=np.float)
        item['visual'] = Image.fromarray(img)

        if self.transform is not None:
            item['visual']  = self.transform(item['visual'])

        _annotation = self.annotations[index] # read in the annotations
        gt_boxes_object = np.zeros((len(_annotation['objects']), 5)) # placeholder for bbox's
        # scale and store the gt bbox's coordinates and class
        gt_boxes_object[:, 0:4] = np.array([obj['bbox'] for obj in _annotation['objects']], dtype=np.float) * im_scale
        gt_boxes_object[:, 4]   = np.array([obj['class'] for obj in _annotation['objects']])
        item['objects'] = gt_boxes_object

        # calculate the RPN targets - need to mess it with
        # initial analysis - computations is w.r.t to anchors thats why we need to assign
        # the gt boxes to anchors!
        if self.cfg_key == 'train': 
            item['rpn_targets']['object'] = anchor_target_layer(item['visual'], 
                                                                gt_boxes_object, 
                                                                item['image_info'],
                                                                self._feat_stride, self._rpn_opts['object'],
                                                                mappings = self._rpn_opts['mappings'])

        gt_relationships = np.zeros([len(_annotation['objects']), (len(_annotation['objects']))], dtype=np.long)
        for rel in _annotation['relationships']:
            gt_relationships[rel['sub_id'], rel['obj_id']] = rel['predicate']
        item['relations'] = gt_relationships

        return item

    @staticmethod
    def collate(items):
        """
            Used by the data loader for specifying how the different items (fetched using the indices) will be grouped together
        """
        batch_item = {}
        
        for key in items[0]:
            if key == 'visual':
                batch_item[key] = [x[key].unsqueeze(0) for x in items]
            elif key == 'rpn_targets':
                batch_item[key] = {}
                for subkey in items[0][key]:
                    batch_item[key][subkey] = [x[key][subkey] for x in items]
            elif items[0][key] is not None:
                batch_item[key] = [x[key] for x in items]

        return batch_item


    def __len__(self):
        return len(self.annotations)

    @property
    def voc_size(self):
        return len(self.idx2word)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(i)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = self.annotations[index]['path']
        image_path = osp.join(self._data_path, file_name)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _image_resize(self, im, target_size, max_size):
        """
            Builds an input blob from the images in the roidb at the specified scales
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        
        # prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        return im, im_scale

    @property
    def name(self):
        return self._name

    @property
    def num_object_classes(self):
        return len(self._object_classes)

    @property
    def num_predicate_classes(self):
        return len(self._predicate_classes)

    @property
    def object_classes(self):
        return self._object_classes

    @property
    def predicate_classes(self):
        return self._predicate_classes
