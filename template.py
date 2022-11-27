#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/Castella99/OSSProject2

import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import uniform, randint

def load_dataset(dataset_path): # csv 파일 읽기
	#To-Do: Implement this function
	dataset = pd.read_csv(dataset_path)
	return dataset
	
def dataset_stat(dataset_df): # 데이터셋 스탯 확인
	#To-Do: Implement this function
	# feature 개수, class 0 / class 1 비율
	dataset_ft = dataset_df.iloc[:, :-1]
	dataset_tg = dataset_df.iloc[:,-1]
	dataset_ft_size = dataset_ft.shape[1]
	dataset_tg_0 = dataset_tg.loc[dataset_tg==0]
	dataset_tg_0 = dataset_tg_0.size
	dataset_tg_1 = dataset_tg.loc[dataset_tg==1]
	dataset_tg_1 = dataset_tg_1.size
	return dataset_ft_size, dataset_tg_0, dataset_tg_1
	

def split_dataset(dataset_df, testset_size): # 데이터 스플릿 함수
	#To-Do: Implement this function
	dataset_np = dataset_df.to_numpy()
	target = dataset_np[:,-1]
	X = dataset_np[:, :-1]
	test_ratio = int(testset_size) / target.size
	x_train, x_test, y_train, y_test = train_test_split(
		X, target, test_size=test_ratio, shuffle=True, stratify=target, random_state=42
	)
	ss = StandardScaler()
	x_train_scaled = ss.fit_transform(x_train)
	x_test_scaled = ss.fit_transform(x_test)
	return x_train_scaled, x_test_scaled, y_train, y_test
 
def decision_tree_train_test(x_train, x_test, y_train, y_test): # 결정트리
	#To-Do: Implement this function
	params = {
    'min_impurity_decrease' : uniform(0.001, 0.001),
    'max_depth' : randint(10, 100),
    'min_samples_split' : randint(2, 100),
    'min_samples_leaf' : randint(1, 100),
    'max_features' : uniform(0.5, 0.5),
    'min_weight_fraction_leaf' : uniform(0.1, 0.4),
	}
	dt = DecisionTreeClassifier(random_state=42)
	rs = RandomizedSearchCV(dt, params, n_iter=100, cv=5, n_jobs=-1, random_state=42)
	rs.fit(x_train, y_train)
	dt = rs.best_estimator_ # best 
	y_pred = dt.predict(x_test)
	acc = dt.score(x_test, y_test)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test): # 랜덤포레스트
	#To-Do: Implement this function
	params = {
    'n_estimators' : randint(10, 1000),
    'max_depth' : randint(10, 100),
    'min_samples_split' : randint(2, 100),
    'min_samples_leaf' : randint(1, 100),
    'min_weight_fraction_leaf' : uniform(0, 0.5),
    'max_features' : uniform(0,1),
    'min_impurity_decrease' : uniform(0.001, 0.001),
    'max_samples' : uniform(0,1),
	}
	rf = RandomForestClassifier(random_state=42)
	rs = RandomizedSearchCV(rf, params, n_iter=5, cv=5, n_jobs=-1, random_state=42)
	rs.fit(x_train, y_train)
	rf = rs.best_estimator_
	y_pred = rf.predict(x_test)
	acc = rf.score(x_test, y_test)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test): # SVM
	#To-Do: Implement this function
	svm = SVC(kernel='rbf')
	svm.fit(x_train, y_train)
	y_pred = svm.predict(x_test)
	acc = svm.score(x_test, y_test)
	prec = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def print_performances(acc, prec, recall): # performance 출력 함수
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

data_path = sys.argv[1]
data_df = load_dataset(data_path) # 데이터 로딩

n_feats, n_class0, n_class1 = dataset_stat(data_df)
print ("Number of features: ", n_feats) # feature 개수
print ("Number of class 0 data entries: ", n_class0) # class 0 개수
print ("Number of class 1 data entries: ", n_class1) # class 1 개수

# 데이터 스플릿
print ("\nSplitting the dataset with the test size of ", sys.argv[2])
x_train, x_test, y_train, y_test = split_dataset(data_df, sys.argv[2])

# 결정 트리
acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
print ("\nDecision Tree Performances")
print_performances(acc, prec, recall)

# 랜덤포레스트
acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
print ("\nRandom Forest Performances")
print_performances(acc, prec, recall)

# 서포트 벡터 머신
acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
print ("\nSVM Performances")
print_performances(acc, prec, recall)