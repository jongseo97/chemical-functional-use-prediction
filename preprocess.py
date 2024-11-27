# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:01:17 2024

@author: jspark
"""
import torch
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from rdkit.Chem.rdmolops import GetMolFrags
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

import numpy as np
import pandas as pd
import pickle

######## GCN dataset ########
# Molecular Graph 생성
class GCNDataset(Dataset):
    
    def __init__(self, max_num_atoms, smiles_list, mf):
        self.max_num_atoms = max_num_atoms
        self.smiles_list = smiles_list
        self.input_feature_list = []
        self.adj_list = []
        self.process_data()
        self.mf = list(map(torch.from_numpy, np.array(mf, dtype = np.float64)))
        
    def process_data(self):
        self.mol_to_graph(self.smiles_list, self.input_feature_list, self.adj_list)

    def mol_to_graph(self, smi_list, feature_list, adj_list):
        max_num_atoms = self.max_num_atoms
        for smiles in smi_list:
            mol = Chem.MolFromSmiles(smiles)
            # 염 제거
            mol = self.remove_salt(mol)
            # partial charge 계산
            AllChem.ComputeGasteigerCharges(mol)
            
            num_atoms = mol.GetNumAtoms()
            
            # Get adjacency matrix
            adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
            
            #degree 높은애들은 계속 더해지니까 normalize 해줌 (DADWH)
            Degree_tilde = 1/np.sqrt(adj.sum(1) + 1) * np.eye(num_atoms)
            norm_adj = Degree_tilde @ adj @ Degree_tilde
            
            # adjacency matrix를 max atom만큼 padding
            padded_adj = np.zeros((max_num_atoms, max_num_atoms))
            padded_adj[:num_atoms, :num_atoms] = norm_adj
            
            #Get atom feature matrix
            feature = []
            for i in range(num_atoms):
                feature.append(self.get_atom_feature(mol, i))
            feature = np.array(feature)
            
            # atom feature matrix를 max atom만큼 padding
            padded_feature = np.zeros((max_num_atoms,feature.shape[1]))
            padded_feature[:num_atoms,:feature.shape[1]] = feature
            
            feature_list.append(torch.from_numpy(padded_feature))
            adj_list.append(torch.from_numpy(padded_adj))

    def onehot_encoding(self, x, allowable_set):
        #Maps inputs not inthe allowable set to the last element
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x==s, allowable_set))
    
    def get_atom_feature(self, m, atom_i):
        # 사용할 atom feature 정의
        atom = m.GetAtomWithIdx(atom_i)
        symbol = self.onehot_encoding(atom.GetSymbol(),['C','N','O','F','Cl','Br','I','S','P','Si', 'Sn', 'Al','ELSE']) # 10
        degree = self.onehot_encoding(atom.GetDegree(), [0,1,2,3,4,5,6])
        charge = self.onehot_encoding(atom.GetFormalCharge(), [-1, 0, 1, 'ELSE'])
        chirality = self.onehot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'])
        Hs = self.onehot_encoding(atom.GetTotalNumHs(), [0,1,2,3,4])
        hybrid = self.onehot_encoding(str(atom.GetHybridization()), ['SP','SP2','SP3','S','SP3D','SP3D2','ELSE'])
        etc = [atom.IsInRing(), atom.GetIsAromatic(), int(float(atom.GetProp('_GasteigerCharge'))>0), atom.GetMass()/100]
        return np.array(symbol+degree+charge+chirality+Hs+hybrid+etc)

    
    def remove_salt(self, mol):
        mols = list(GetMolFrags(mol, asMols=True))
        if mols:
            mols.sort(reverse = True, key = lambda m:m.GetNumAtoms())
            mol = mols[0]
        return mol

    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        sample = dict()
        sample['x'] = self.input_feature_list[idx]
        sample['adj'] = self.adj_list[idx]
        sample['mf'] = self.mf[idx]
        return sample


# rdkit 2D descriptor 생성
def smiles_to_mf(smiles_list):
    
    # mol -> descrptor
    def getMolDescriptors(mol):
        # 사용할 rdkit 2D descriptors (121개)
        mf_names = ["MaxEStateIndex", "MinEStateIndex", "MaxAbsEStateIndex", "MinAbsEStateIndex", "MolWt", "HeavyAtomMolWt", "ExactMolWt", 
        "NumValenceElectrons", "FpDensityMorgan2", "FpDensityMorgan3", "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v", "Chi1", 
        "Chi1n", "Chi1v", "Chi2n", "Chi2v", "Chi3n", "Chi3v", "Chi4n", "Chi4v", "HallKierAlpha", "Kappa1", "Kappa2", "LabuteASA", 
        "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", 
        "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", 
        "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", 
        "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8", "TPSA", "EState_VSA1", "EState_VSA10", 
        "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", 
        "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5", "VSA_EState6", "VSA_EState7", 
        "VSA_EState8", "VSA_EState9", "HeavyAtomCount", "NHOHCount", "NOCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", 
        "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", 
        "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "RingCount", "MolLogP", "MolMR", "fr_Al_OH", 
        "fr_Al_OH_noTert", "fr_Ar_N", "fr_Ar_OH", "fr_C_O", "fr_C_O_noCOO", "fr_NH0", "fr_NH1", "fr_alkyl_halide", "fr_allylic_oxid", 
        "fr_amide", "fr_aniline", "fr_benzene", "fr_bicyclic", "fr_ester", "fr_ether", "fr_halogen", "fr_phenol", "fr_phenol_noOrthoHbond", "fr_unbrch_alkane"]
        mf = []
        for nm, fn in Descriptors._descList:
            if nm in mf_names:
                mf.append(fn(mol))
        return mf        
    
    # 각 분자 smiles에 적용
    mf_list = []
    for smi in smiles_list:
        if pd.isna(smi):
            continue
        
        mol = Chem.MolFromSmiles(smi)
        mf = getMolDescriptors(mol)
        mf_list.append(mf)

    return pd.DataFrame(mf_list)


# training set에 사용한 scaler 불러와서 적용
def scaling_mf(mf, scaler_path):
    scaler = pickle.load(open(scaler_path, 'rb'))
    mf = scaler.transform(mf)
    return mf

# Max Heavy Atoms 128 제한으로 인한 한계 적용
def heavy_limit(dt):
    smiles_list = []
    dt = dt.copy()
    for smiles in dt['SMILES']:
        if pd.isna(smiles):
            smiles_list.append(smiles)
            continue
        if Chem.MolFromSmiles(smiles).GetNumAtoms() > 128:
            smiles_list.append(None)
        else:
            smiles_list.append(smiles)
    dt['SMILES'] = smiles_list
    return dt
            

# 데이터셋 생성 및 torch dataloader로 변경
def preprocessing(smiles_list, mf):
    dataset = GCNDataset(128, smiles_list, mf)
    dataloader = DataLoader(dataset, batch_size = 256)
    return dataloader
    
