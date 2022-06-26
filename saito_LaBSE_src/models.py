import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

import config


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0,  m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.set_margin(m)

    def set_margin(self, m):
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        print(f"Set margin parmas: m={m:.3f}, cos_m={self.cos_m:.3f}, sim_m={self.sin_m:.3f}, th={self.th:.3f}, mm={self.mm:.3f}")

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=config.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class FSNet(nn.Module):
    def __init__(self, model_name):
        super(FSNet, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, config=self.config)
        # self.embedding = nn.Linear(self.config.hidden_size + 2, embedding_size)

        self.fc = nn.Linear(self.bert_model.config.hidden_size, config.fc_dim)
        self.bn = nn.BatchNorm1d(config.fc_dim)
        self._init_params()

        self.margin = ArcMarginProduct(
            config.fc_dim,
            config.n_classes,
            s=config.s, 
            m=config.margin, 
            easy_margin=config.easy_margin,
            ls_eps=config.ls_eps
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, ids, mask, lat, lon, coord_x, coord_y, coord_z, labels):
        feature = self.extract_feature(ids, mask, lat, lon, coord_x, coord_y, coord_z)
        output = self.margin(feature, labels)

        return output
    
    def extract_feature(self, input_ids, attention_mask, lat, lon, coord_x, coord_y, coord_z):
        x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = x.last_hidden_state

        x = torch.sum(
            x * attention_mask.unsqueeze(-1), dim=1, keepdim=False
        )
        x = x / torch.sum(attention_mask, dim=-1, keepdim=True)

        x = self.fc(x)
        x = self.bn(x)

        return x
    

class FSMultiModalNet(nn.Module):
    def __init__(self, model_name, num_features=3):
        super(FSMultiModalNet, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.bert_model.gradient_checkpointing_enable() 
        # self.embedding = nn.Linear(self.config.hidden_size + 2, embedding_size)

        print(self.bert_model.config.hidden_size + num_features)

        self.fc = nn.Linear(self.bert_model.config.hidden_size + num_features, config.fc_dim)
        self.bn = nn.BatchNorm1d(config.fc_dim)
        self._init_params()

        self.margin = ArcMarginProduct(
            config.fc_dim,
            config.n_classes,
            s=config.s, 
            m=config.m_start, 
            easy_margin=config.easy_margin,
            ls_eps=config.ls_eps
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def update_margin(self, m):
        self.margin.set_margin(m)

    def forward(self, ids, mask, lat, lon, coord_x, coord_y, coord_z, labels):
        feature = self.extract_feature(ids, mask, lat, lon, coord_x, coord_y, coord_z)
        output = self.margin(feature, labels)

        return output
    
    def extract_feature(self, input_ids, attention_mask, lat, lon, coord_x, coord_y, coord_z):
        x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)

        # x = torch.cat([x, lat.view(-1, 1), lon.view(-1, 1)], axis=1)
        x = torch.cat([x, coord_x.view(-1, 1), coord_y.view(-1, 1), coord_z.view(-1, 1)], axis=1)

        x = self.fc(x)
        x = self.bn(x)

        return x
