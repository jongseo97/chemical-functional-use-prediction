# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:04:31 2024

@author: jspark
"""

from model import GCNNet_Classification

import torch
import numpy as np

# model 생성 및 파라미터 적용
def load_model(model_path):
    model = GCNNet_Classification(128, 3, 4, 43, 64, 512, 36, True, True, 0.5, True, 121, 'mean')
    
    # CPU/GPU 확인
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 파라미터 적용
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    #GPU 사용
    if torch.cuda.is_available():
        model.cuda()
        
    #파라미터 고정
    model.eval()
    
    return model


# 모델과 데이터를 통해 예측
def make_prediction(model, dataloader):
    # 파라미터 고정
    with torch.no_grad():
        # preds=확률, y=라벨
        preds = []
        y = []
        for i_batch, batch in enumerate(dataloader):
            # CPU/GPU 확인
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            x = batch['x'].to(device).float()
            adj = batch['adj'].to(device).float()
            mf = batch['mf'].to(device).float()
            
            pred = model(x, adj, mf).squeeze(-1)
            
            preds.append(torch.sigmoid(pred.cpu()))
            y.append(torch.sigmoid(pred.cpu())>=0.5)
            
        preds = np.concatenate(preds)
        y = np.concatenate(y)
        
    return preds, y

