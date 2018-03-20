# -*- enconding:utf-8 -*-
import numpy as np 
import pandas as pd 
from frame import *
from model import *
from function import *
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
#import seaborn as sns
import copy


# Light GBM
lgb_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'rmse'},
    'num_leaves': 128,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 6,
    'verbosity': 2,
    'tree_learner':'feature',
    'min_sum_hessian_in_leaf':0.1
}

# XGBoost parameters 
xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'stratified':True,
    #'scale_pos_weights ':0,
    'max_depth':5,
    'min_child_weight':1,
    'gamma':1,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'lambda':1,
    
    'eta':0.1,
    'seed':20,
    'silent':1,
    'eval_metric':'rmse'
}

config = {
    'trick':True,
    'user':True,
    'goods_type':['spu_id','brand_id','cat_id'],
    'spu':True,
    'brand':True,
    'cat':True,
    'index':True,
    'user2':True,
}


if __name__ == '__main__':
	Log('start preprocess...')
	preprocess()
	Log('start get getWindowFeature..')
	df = pd.read_csv('../data/local_train.csv')
	df2 = pd.read_csv('../data/online_test.csv')
	df2['week'] = -1
	df = df.drop('date',axis=1)
	goods = pd.read_csv('../data/goods.csv')
	df = df2.append(df).reset_index(drop=True)
	df = pd.merge(df,goods,on='spu_id',how='left')
	Log('getWindowFeature 9 ..')
	features = getWindowFeature(df,[-1,0,1,2],9,config)
	config['trick'] = False
	config['index'] = False
	Log('getWindowFeature 4 ..')
	features = getWindowFeature(df,[-1,0,1,2],4,config)
	Log('getWindowFeature 1 ..')
	features = getWindowFeature(df,[-1,0,1,2],1,config)

	getFeature([1,2],0,np.arange(50))
	getFeature([0,1],-1, np.arange(50))

	SEED = 1024
	xgb = XgbWrapper(seed=SEED, params=xgb_params, rounds = 550, has_weight = False)
	lgb = LgbWrapper(seed=SEED, params=lgb_params, rounds = 80, has_weight = False)
	

	for s in np.arange(50):
		Log('local-online model-%d'%s)
		train_x, train_y, test_x = readLocalOnlineFeature(s)
		Log(train_x.shape)
		Log(test_x.shape)
		Log(train_y.shape)
		frame1 = MLframe(train_x, train_y, test_x)
		clf, predict = frame1.train_predict(xgb)
		result = storeResult(predict,'final-local-online-xgb-'+str(s), 'local-online-xgb')
		clf, predict = frame1.train_predict(lgb)
		result = storeResult(predict,'final-local-online-lgb-'+str(s), 'local-online-lgb')


	for s in np.arange(50):
		Log('online-online model-%d'%s)
		train_x, train_y, test_x = readOnlineOnlineFeature(s)
		Log(train_x.shape)
		Log(test_x.shape)
		Log(train_y.shape)
		frame1 = MLframe(train_x, train_y, test_x)
		clf, predict = frame1.train_predict(xgb)
		result = storeResult(predict,'final-online-online-xgb-'+str(s), 'online-online-xgb')
		clf, predict = frame1.train_predict(lgb)
		result = storeResult(predict,'final-online-online-lgb-'+str(s), 'online-online-lgb')


	
	day = np.array(range(50))
	blend(day,'local-online-xgb','xgb-local-online-blend-%d'%len(day))
	blend(day,'online-online-xgb','xgb-online-online-blend-%d'%len(day))

	blend(day,'local-online-lgb','lgb-local-online-blend-%d'%len(day))
	blend(day,'online-online-lgb','lgb-online-online-blend-%d'%len(day))

	final_blend([0.4,0.3,0.2,0.1])
	
	
	
	

