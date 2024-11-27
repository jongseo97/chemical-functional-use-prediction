1. pytorch 설치
2. rdkit 버전 >= 2022.09.4
3. 파일 description
- preprocess.py : 전처리에 사용되는 함수 모음
	1-1. GCNDataset : smiles_list를 molecular graph로 변환한 뒤 pytorch Dataset으로 변환
	1-2. smiles_to_mf : smiles_list를 molecular feature data frame으로 변환
	1-3. scaling_mf : scaler 파라미터를 불러와서 molecular feature scaling
	1-4. preprocessing : smiles_list를 1-1 함수 사용, GCNDataset으로 변환. 이후 pytorch 학습/예측에 사용되는 DataLoader로 변환

- model.py : 예측 모델에 사용되는 Class 모음
	GCNLayer, Readout, Predictor 각 블록을 활용하여 GCNNet_Classification 모델 구성

- predict.py : 예측 결과 생성에 사용되는 함수 모음
	2-1. load_model : 모델 생성 및 파라미터 적용
	2-2. make_prediction : 2-1 모델과 1-4 DataLoader 불러와서 예측 결과 생성

- utils.py : 이외 함수 모음 (제품군 분류하여 연결시키는 함수도 추가 예정)
	3-1. pred_to_function : True, False로 나온 predicted label을 기능명으로 매칭
	3-2. pred_proba_to_df : 확률값으로 나온 predicted value를 data frame으로 정리
	3-3. prediction_to_dataframe : 3-1과 3-2 함수를 사용, output 파일 생성
	3-4. product_category_matching : 예측된 기능들이 포함하고 있는 제품군을 연결하여 data frame에 column 추가

- main.py : 메인 구동 파일
	4. main : 스크립트가 모여있는 파일 위치, 예측할 smiles list 파일 위치, output 파일명을 받아 1, 2, 3 함수를 사용하여 예측 수행

- parameters : 모델 파라미터 모음
	scaler.sav : train data의 molecular feature를 스케일링 할 때 사용한 scaler 파라미터
	Functions_weights.pth : 기능 예측을 수행한 모델의 파라미터


	