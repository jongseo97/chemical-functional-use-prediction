# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:57:29 2024

@author: jspark
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


######## GCN model ########

#GCN model blocks
class GCNLayer(torch.nn.Module):
    def __init__(self, n_atom, in_dim, out_dim, use_bn):
        super(GCNLayer, self).__init__()
        self.n_atom = n_atom
        self.in_dim = in_dim
        self.out_dim = out_dim        
        self.use_bn = use_bn
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.bn = nn.BatchNorm1d(self.n_atom)
        
    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        if self.use_bn:
            x = self.bn(x)
        retval = F.relu(x)
        return retval

class Readout(torch.nn.Module):
    def __init__(self, in_dim, out_dim, readout):
        super(Readout, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.readout = readout
        
    def forward(self, x):
        x = self.linear(x)
        if self.readout == 'sum':
            x = x.sum(1)
        else:
            x = x.mean(1)
        retval = F.relu(x)
        return retval

class Predictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_dropout, drop_rate):
        super(Predictor, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_dropout = use_dropout
        self.drop_rate = drop_rate
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout(self.drop_rate)
        self.bn = nn.BatchNorm1d(self.out_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        retval = F.relu(x)
        if self.use_dropout:
            retval = self.dropout(retval)
        return retval
    
    
#Classification model structure
class GCNNet_Classification(torch.nn.Module):
    def __init__(self, max_atoms, n_conv, n_MLP, n_atom_feature, n_conv_feature, n_feature, n_out, use_bn, use_dropout, drop_rate, use_mf, n_mf, readout):
        super(GCNNet_Classification, self).__init__()
        self.use_mf = use_mf        
        self.embedding = nn.Linear(n_atom_feature, n_conv_feature)
        
        GCN_list = []
        for _ in range(n_conv):
            GCN_list.append(GCNLayer(max_atoms, n_conv_feature, n_conv_feature, use_bn))
        self.GCN_list = nn.ModuleList(GCN_list)
        
        self.readout = Readout(n_conv_feature, n_feature, readout)
        if use_mf:
            in_feature = n_feature + n_mf        
        else:
            in_feature = n_feature        
        
        MLP_list = []
        for i in range(n_MLP):
            if i==0:
                MLP_list.append(Predictor(in_feature, n_feature, use_dropout, drop_rate))
            else:
                MLP_list.append(Predictor(n_feature, n_feature, use_dropout, drop_rate))
        self.MLP_list = nn.ModuleList(MLP_list)
        
        self.fc = nn.Linear(n_feature, n_out)
        self.sigmoid = nn.Sigmoid()
                
    def forward(self, x, adj, mf):
        x = self.embedding(x)
        
        for layer in self.GCN_list:
            x = layer(x, adj)
        x = self.readout(x)
        if self.use_mf:
            x = torch.cat([x, mf], dim=1)
                
        for layer in self.MLP_list:
            x = layer(x)
        retval = self.fc(x)
                    
        return retval