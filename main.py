# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:14:48 2024

@author: jspark
"""
import os
os.chdir(r'C:\Users\user\Desktop\1\Modeling\17. 대체물질 탐색 알고리즘\FunctionUse\model')

import pandas as pd

from preprocess import smiles_to_mf, scaling_mf, preprocessing, heavy_limit
from predict import load_model, make_prediction
from utils import pred_to_function, pred_proba_to_df, prediction_to_dataframe, product_category_matching
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# dir_path : script 파일 위치
# input_path : 예측할 SMILES list 파일이름 (csv or xlsx)
# output_path : output 파일 이름 (csv or xlsx)
# save_proba : True=예측 raw 결과 함께 출력 / False=기능명만 출력
def main(dir_path, input_path, output_path='output.xlsx', save_proba = False):
    # model과 scaler parameter는 항상 dir_path의 parameters 폴더 내에 있어야함
    model_path = f'{dir_path}/parameters/Functions_weights.pth'
    scaler_path = f'{dir_path}/parameters/scaler.sav'
    
    # input = csv or xlsx
    if '.csv' in input_path:
        dt = pd.read_csv(input_path)
    elif '.xlsx' in input_path:
        dt = pd.read_excel(input_path)
    else:
        raise Exception('Input type is .csv or .xlsx')
        
    # heavy atom 128 제한 적용    
    new_dt = heavy_limit(dt)
    
    # rdkit 2D molecular feature 생성 및 scaling
    smiles_list = new_dt['SMILES']
    smiles_list = smiles_list.dropna()        
    
    
    mf = smiles_to_mf(smiles_list)
    mf = scaling_mf(mf, scaler_path)
    
    # smiles_list와 molecular feature로 예측 데이터셋 생성
    dataloader = preprocessing(smiles_list, mf)
    
    # 모델 파라미터 로드, 예측
    model = load_model(model_path)
    pred_proba, pred = make_prediction(model, dataloader)
    
    # 예측 결과 -> 데이터프레임 정리
    df = prediction_to_dataframe(smiles_list, pred, pred_proba, save_proba)
    df = product_category_matching(df, dir_path)
   
    dt['Functions'] = None
    dt.loc[new_dt['SMILES'].notna(), 'Functions'] = df['Functions'].values
    
    dt['Product Family'] = None
    dt.loc[new_dt['SMILES'].notna(), 'Product Family'] = df['Product Family'].values
    
    # output = csv or xlsx
    if '.csv' in output_path:
        dt.to_csv(output_path, index=False)
    elif '.xlsx' in output_path:
        dt.to_excel(output_path, index=False)
    else:
        raise Exception('output type is .csv or .xlsx')
        
    return dt


if __name__ == '__main__':
    
    dir_path = r'C:\Users\user\Desktop\1\Modeling\17. 대체물질 탐색 알고리즘\FunctionUse\model'
    input_path = r'C:\Users\user\Desktop\1\DB\20240927_생활화학제품_저작권_DB\20240829_생활화학제품_DB_GHS_smiles_fuse.xlsx'
    input_path = r'C:\Users\user\Desktop\1\DB\20241029_어린이제품_저작권_DB\과정 파일\어린이제품_GHS.xlsx'
    output_path = r'C:\Users\user\Desktop\1\DB\20240927_생활화학제품_저작권_DB\20240927_생활화학제품_DB_GHS_smiles_fuse.xlsx'
    output_path = r'C:\Users\user\Desktop\1\DB\20241029_어린이제품_저작권_DB\과정 파일\어린이제품_FUse.xlsx'
    results = main(dir_path, input_path, output_path, save_proba = False)





import torch
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# 학습된 모델의 출력 계층 가중치 추출
fc_weights = model.fc.weight.detach().cpu().numpy()  # (36 x 512)

# 기능 간 코사인 유사도 계산
cosine_similarity = np.dot(fc_weights, fc_weights.T) / (
    np.linalg.norm(fc_weights, axis=1)[:, None] * np.linalg.norm(fc_weights, axis=1)
)

# 임계값 적용
threshold = 0.5  # 상관계수 임계값
filtered_similarity = np.where((cosine_similarity > threshold) | (cosine_similarity < -threshold), cosine_similarity, 0)

# 히트맵 시각화
sns.heatmap(filtered_similarity, annot=False, xticklabels=[f'F{functions[i]}' for i in range(36)], 
            yticklabels=[f'F{functions[i]}' for i in range(36)], cmap="coolwarm")
plt.title("Feature Correlation Heatmap (Filtered)")
plt.show()

sns.heatmap(cosine_similarity, cmap="coolwarm",
            xticklabels=[f'{functions[i]}' for i in range(fc_weights.shape[0])],
            yticklabels=[f'{functions[i]}' for i in range(fc_weights.shape[0])])
plt.title("Class-to-Class Cosine Similarity")
plt.xlabel("Classes")
plt.ylabel("Classes")
plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns

# 코사인 유사도를 기반으로 거리 행렬 계산
distance_matrix = 1 - cosine_similarity  # 1-코사인 유사도

# 계층적 클러스터링
linkage_matrix = linkage(squareform(distance_matrix), method="ward")

# 클러스터링 히트맵
sns.clustermap(cosine_similarity, cmap="coolwarm", row_linkage=linkage_matrix, col_linkage=linkage_matrix,
               xticklabels=[f'F{i}' for i in range(36)], yticklabels=[f'F{i}' for i in range(36)])
plt.title("Clustered Feature Correlation Heatmap")
plt.show()


import networkx as nx

# 그래프 생성
G = nx.Graph()
threshold = 0.6  # 네트워크에 포함할 상관관계 임계값

for i in range(len(cosine_similarity)):
    for j in range(i+1, len(cosine_similarity)):
        if cosine_similarity[i, j] > threshold:  # 양의 상관관계만 추가
            G.add_edge(f'{functions[i]}', f'{functions[j]}', weight=cosine_similarity[i, j])

# 그래프 시각화
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
nx.draw_networkx_edges(G, pos, width=1.0, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
plt.title("Feature Correlation Network")
plt.show()


sns.heatmap(filtered_similarity, annot=False, xticklabels=label_names, yticklabels=label_names, cmap="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()

plt.hist(cosine_similarity.flatten(), bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Feature Correlations")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()

plt.hist(cosine_similarity.flatten(), bins=20, color="blue", alpha=0.7)
plt.title("Distribution of Class-to-Class Cosine Similarity")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.show()