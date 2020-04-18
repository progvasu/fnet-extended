import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Dumplicate_Removal(nn.Module):
    def __init__(self, opts):
        super(Dumplicate_Removal, self).__init__()
        self.opts = opts
        self.relation_transform = Relation_Module(
            self.opts['dim_mm'],
            self.opts['dim_mm'],
            self.opts['dim_mm'] // 2,
            geometry_trans=self.opts.get('geometry', 'Geometry_Transform_v2')
        )
        self.transform_visual = nn.Linear(self.opts['dim_ho'], self.opts['dim_mm'])
        self.rank_embeddings = nn.Embedding(256, self.opts['dim_mm'])
        self.transform_rescore = nn.Linear(self.opts['dim_mm'], 1)

    def forward(self, feature_obj, highest_prob, rois_obj):
        assert highest_prob.size(0) <= self.rank_embeddings.num_embeddings
        if isinstance(highest_prob, Variable):
            highest_prob = highest_prob.data
        _, rank = torch.sort(highest_prob, descending=True, dim=0)
        rank = Variable(rank)
        feature_rank = self.rank_embeddings(rank)
        feature_obj = self.transform_visual(feature_obj)
        feature_visual = feature_rank + feature_obj
        feature_visual = self.relation_transform(feature_visual, rois_obj)
        reranked_score = self.transform_rescore(F.relu(feature_visual, inplace=True)) 
        reranked_score = torch.sigmoid(reranked_score)

        return reranked_score


class Relation_Module(nn.Module):
      def __init__(self, dim_v, dim_o, dim_mm, geometry_trans='Geometry_Transform_v2'):
            super(Relation_Module, self).__init__()
            self.dim_key = dim_mm
            self.transform_key = nn.Linear(dim_v, dim_mm)
            self.transform_query = nn.Linear(dim_v, dim_mm)
            self.transform_visual = nn.Linear(dim_v, dim_o)
            self.transform_geometry = Geometry_Transform_v2(dim_mm)

      def forward(self, feature_visual, rois):
            feature_visual = nn.functional.relu(feature_visual)
            feature_key = self.transform_key(feature_visual)
            feature_query = self.transform_query(feature_visual)
            feature_visual = self.transform_visual(feature_visual)

            visual_weight = (feature_query.unsqueeze(0) * feature_key.unsqueeze(1)).sum(dim=2, keepdim=False) / np.sqrt(self.dim_key)
            geometry_weight = self.transform_geometry(rois)

            attention = visual_weight.exp() * geometry_weight
            for i in range(attention.size(0)):
                  attention[i, i] = 0
            attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-10)
            feature_out = torch.sum(attention.unsqueeze(2) * feature_visual.unsqueeze(0), dim=1, keepdim=False)

            return feature_out


def geometry_transform(rois_keys, rois_queries=None):
      if rois_queries is None:
            rois_queries = rois_keys
      if isinstance(rois_keys, Variable): # transform to Tensor
            rois_keys = rois_keys.data
            rois_queries = rois_queries.data
      if rois_keys.size(1) == 5: # Remove the ID
            rois_keys = rois_keys[:, 1:]
            rois_queries = rois_queries[:, 1:]

      assert rois_keys.size(1) == 4
      # keys
      w_keys = (rois_keys[:, 2] - rois_keys[:, 0] + 1e-10).unsqueeze(1)
      h_keys = (rois_keys[:, 3] - rois_keys[:, 1] + 1e-10).unsqueeze(1)
      x_keys = ((rois_keys[:, 2] + rois_keys[:, 0]) / 2).unsqueeze(1)
      y_keys = ((rois_keys[:, 3] + rois_keys[:, 1]) / 2).unsqueeze(1)
      # queries
      w_queries = (rois_queries[:, 2] - rois_queries[:, 0] + 1e-10).unsqueeze(0)
      h_queries = (rois_queries[:, 3] - rois_queries[:, 1] + 1e-10).unsqueeze(0)
      x_queries = ((rois_queries[:, 2] + rois_queries[:, 0]) / 2).unsqueeze(0)
      y_queries = ((rois_queries[:, 3] + rois_queries[:, 1]) / 2).unsqueeze(0)

     # slightly different from [Relation Networks for Object Detection]
      geometry_feature = torch.stack(
          [(x_keys - x_queries).abs() / w_keys,
           (y_keys - y_queries).abs() / h_keys,
           w_keys / w_queries,
           h_keys / h_queries,], dim=2)

      geometry_log = geometry_feature.log()
      geometry_log[geometry_feature == 0] = 0

      return geometry_log

def positional_encoding(position_mat, dim_output, wave_length=1000):
      assert dim_output % 8 == 0, "[dim_output] is expected to be an integral multiple of 8"
      position_enc = torch.Tensor([np.power(wave_length, 8.*i/dim_output) for i in range(dim_output / 8)]).view(1, 1, 1, -1).type_as(position_mat)
      position_enc = position_mat.unsqueeze(-1) * 100 / position_enc
      position_enc = torch.cat([torch.sin(position_enc), torch.cos(position_enc)], dim=3)
      position_enc = position_enc.view(position_enc.size(0), position_enc.size(1), -1)

      return position_enc 

class Geometry_Transform_v2(nn.Module):
      def __init__(self, dim_mm):
            super(Geometry_Transform_v2, self).__init__()
            self.transform_geometry = nn.Sequential(
                                            nn.Linear(dim_mm, 1),
                                            nn.ReLU(),)
            self.dim_mm = dim_mm

      def forward(self, rois_keys, rois_queries=None):
            position_mat = geometry_transform(rois_keys, rois_queries)
            geometry_weight = positional_encoding(position_mat, self.dim_mm)
            geometry_weight = Variable(geometry_weight, requires_grad=True)  
            geometry_weight = self.transform_geometry(geometry_weight).squeeze(2)
            return geometry_weight