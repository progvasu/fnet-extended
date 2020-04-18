import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.dumplicate_removal import Relation_Module
from lib.network import GroupDropout


class FS(nn.Module):
	def __init__(self, opts):
		super(FS, self).__init__()

		# to transform the attentioned features
		self.transform_object2region = nn.Sequential(
											nn.ReLU(),
											nn.Conv2d(opts['dim_ho'], opts['dim_hr'], kernel_size=1, padding=0, bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		
		self.transform_region2object = nn.Sequential(
											nn.ReLU(),
											nn.Linear(opts['dim_hr'], opts['dim_ho'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)

		# attention computation
		self.att_region2object_obj = nn.Sequential(
											nn.ReLU(),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.att_region2object_reg = nn.Sequential(
											nn.ReLU(),
											nn.Conv2d(opts['dim_hr'], opts['dim_mm'], kernel_size=1, padding=0, bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		
		self.att_object2region_reg = nn.Sequential(
											nn.ReLU(),
											nn.Conv2d(opts['dim_hr'], opts['dim_mm'], kernel_size=1, padding=0, bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		self.att_object2region_obj = nn.Sequential(
											nn.ReLU(),
											nn.Linear(opts['dim_ho'], opts['dim_mm'], bias=opts['use_bias']),
											GroupDropout(p=opts['dropout'], inplace=True),)
		
		self.opts = opts

	@staticmethod
	def _attention_merge(reference, query, features):
		'''
            Input:
                reference: vector [C] | [C x H x W]
                query: batched vectors [B x C] | [B x C x 1 x 1]
            Output:
                merged message vector: [C] or [C x H x W]
		'''
		C = query.size(1)
		assert query.size(1) == reference.size(0)
		similarity = torch.sum(query * reference.unsqueeze(0), dim=1, keepdim=True) / np.sqrt(C + 1e-10) #  follow operations in [Attention is all you need]
		prob = F.softmax(similarity, dim=0)
		weighted_feature = torch.sum(features * prob, dim=0, keepdim=False)
		return weighted_feature

	def region_to_object(self, feat_obj, feat_region, select_mat):
		feat_obj_att = self.att_region2object_obj(feat_obj)
		feat_reg_att = self.att_region2object_reg(feat_region).transpose(1, 3) # transpose the [channel] to the last
		feat_region_transposed = feat_region.transpose(1, 3)
		C_att = feat_reg_att.size(3)
		C_reg = feat_region_transposed.size(3)

		feature_data = []
		transfer_list = np.where(select_mat > 0)
		for f_id in range(feat_obj.size(0)):
			assert len(np.where(select_mat[f_id, :] > 0)[0]) > 0, "Something must be wrong. Please check the code."
			source_indices = transfer_list[1][transfer_list[0] == f_id]
			source_indices = Variable(torch.from_numpy(source_indices).type(torch.cuda.LongTensor), requires_grad=False)
			feat_region_source = torch.index_select(feat_region_transposed, 0, source_indices)
			feature_data.append(self._attention_merge(feat_obj_att[f_id],
								torch.index_select(feat_reg_att, 0, source_indices).view(-1, C_att),
								feat_region_source.view(-1, C_reg),))
		return torch.stack(feature_data, 0)

	def object_to_region(self, feat_region, feat_obj, select_mat):
		'''
		    INPUT:
                feat_region: B x C x H x W
                feat_obj: B x C
		'''
		feat_reg_att = self.att_object2region_reg(feat_region)
		feat_obj_att = self.att_object2region_obj(feat_obj).view(feat_obj.size(0), -1, 1, 1)
		feat_obj = feat_obj.view(feat_obj.size(0), -1, 1, 1)
		feature_data = []
		transfer_list = np.where(select_mat > 0)
		for f_id in range(feat_region.size(0)):
			assert len(np.where(select_mat[f_id, :] > 0)[0]) > 0, "Something must be wrong!"
			source_indices = transfer_list[1][transfer_list[0] == f_id]
			source_indices = Variable(torch.from_numpy(source_indices).type(torch.cuda.LongTensor), requires_grad=False)
			feature_data.append(self._attention_merge(feat_reg_att[f_id],
								torch.index_select(feat_obj_att, 0, source_indices),
								torch.index_select(feat_obj, 0, source_indices)))
		return torch.stack(feature_data, 0)


class factor_updating_structure(FS):
    def __init__(self, opts):
        super(factor_updating_structure, self).__init__(opts)

        kernel_size = opts.get('kernel_size', 1)
        assert kernel_size % 2, 'Odd kernel size required.'

        # To transform the attentioned features
        self.transform_object2object = Relation_Module(opts['dim_ho'], opts['dim_ho'], opts['dim_ho'] // 2, 
                                        geometry_trans=self.opts.get('geometry', 'Geometry_Transform_v2'))


    def forward(self, feature_obj, feature_region, mat_object, mat_region, object_rois, region_rois):
        feature_region2object = self.region_to_object(feature_obj, feature_region, mat_object)

        # transform the features
        out_feature_object = feature_obj + self.transform_region2object(feature_region2object) \
                            + self.transform_object2object(feature_obj, object_rois)

        # gather the attentioned features
        feature_object2region = self.object_to_region(feature_region, feature_obj, mat_region)
        
        # transform the features
        out_feature_region = feature_region + self.transform_object2region(feature_object2region)

        return out_feature_object, out_feature_region