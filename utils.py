# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:28:29 2024

@author: jspark
"""

import pandas as pd
import numpy as np

# 0, 1, 0, 1 로 나오는 예측 값을 기능명과 매칭
def pred_to_function(pred):
    
    functions = ['Adhesion/cohesion promoter', 'Anti-static agent',
           'Antioxidant', 'Binder', 'Biocide', 'Catalyst', 'Chelating agent',
           'Chemical reaction regulator', 'Cleaning agent',
           'Degradant/impurity (EPA)', 'Deodorizer', 'Dispersing agent', 'Dye',
           'Emulsifier', 'Film former', 'Flame retardant',
           'Flavouring and nutrient', 'Foamant', 'Fragrance', 'Hardener',
           'Humectant', 'Pharmaceutical (EPA)', 'Pigment', 'Plasticizer',
           'Preservative', 'Processing aids not otherwise specified',
           'Softener and conditioner', 'Solubility enhancer', 'Solvent',
           'Stabilizing agent', 'Surfactant (surface active agent)',
           'Thickening agent', 'UV stabilizer', 'Viscosity modifier',
           'Wetting agent (non-aqueous)', 'pH regulating agent']

    # 예측값이 True or 1인 index에 기능명 매칭    
    pred_funcs = []
    for p in pred:
        true_indices = np.where(p==True)[0] 
        true_funcs = [functions[idx] for idx in true_indices]
        pred_funcs.append('; '.join(true_funcs))
    
    return pred_funcs

# 확률로 나오는 예측 raw 결과를 data frame 형식으로 정리
def pred_proba_to_df(pred_proba):
    
    functions = ['Adhesion/cohesion promoter', 'Anti-static agent',
           'Antioxidant', 'Binder', 'Biocide', 'Catalyst', 'Chelating agent',
           'Chemical reaction regulator', 'Cleaning agent',
           'Degradant/impurity (EPA)', 'Deodorizer', 'Dispersing agent', 'Dye',
           'Emulsifier', 'Film former', 'Flame retardant',
           'Flavouring and nutrient', 'Foamant', 'Fragrance', 'Hardener',
           'Humectant', 'Pharmaceutical (EPA)', 'Pigment', 'Plasticizer',
           'Preservative', 'Processing aids not otherwise specified',
           'Softener and conditioner', 'Solubility enhancer', 'Solvent',
           'Stabilizing agent', 'Surfactant (surface active agent)',
           'Thickening agent', 'UV stabilizer', 'Viscosity modifier',
           'Wetting agent (non-aqueous)', 'pH regulating agent']
    
    pred_proba = pd.DataFrame(pred_proba)
    pred_proba.columns = functions
    return pred_proba
    
# 위 두 함수를 사용하여 output 파일 생성
def prediction_to_dataframe(smiles_list, pred, pred_proba, save_proba):
    pred_funcs = pred_to_function(pred)
    df = pred_proba_to_df(pred_proba)
    
    cols = list(df.columns)
    df['SMILES'] = smiles_list
    df['Functions'] = pred_funcs
    
    # 확률 출력 False인 경우 기능명만 출력
    if save_proba:
        cols = ['SMILES', 'Functions'] + cols
    else:
        cols = ['SMILES', 'Functions']
    df = df[cols]
    return df

def product_category_matching(df, dir_path):
    
    matching_df = pd.read_excel(f'{dir_path}/기능-제품군 매칭.xlsx')
    products_df = pd.read_excel(f'{dir_path}/제품군 설명.xlsx')
    matching_dict = {row['Function']:row['Categories'] for _, row in matching_df.iterrows()}
    products_dict = {row['분류']:row['이름'] for _, row in products_df.iterrows()}   
    
    functions = df['Functions']
    
    matching_list = []
    products_list = []
    for funcs in functions:
        if funcs == '':
            matching_list.append('None')
            products_list.append('None')
            continue
            
        matching_temp = []
        funcs_list = funcs.split('; ')
        for func in funcs_list:
            matching = matching_dict[func]
            matching = matching.split(', ')
            matching_temp += matching
            
        matching_temp = list(set(matching_temp))
        matching_temp = list(map(int, matching_temp))
        matching_temp.sort()
        products = [products_dict[int(m)] for m in matching_temp]        
        
        products_list.append('; '.join(products))
    
    cols = list(df.columns)
    df['Product Family'] = products_list
    cols.insert(2, 'Product Family')
    df = df[cols]
    print(df)
    return df
        
        
        
        
        
        
    
    