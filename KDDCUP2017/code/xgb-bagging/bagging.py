# -*- encoding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb

FEATUREPATH = '../../feature/xgb-1/'
RESULTPATH = '../../result/'
RESULTNAME = 'xgb-bagging.csv'

train_stat = pd.read_csv(FEATUREPATH+'trainFeature_stat.csv')
train_volume = pd.read_csv(FEATUREPATH+'time_volume_features_train.csv')

test_stat = pd.read_csv(FEATUREPATH+'testFeature_stat.csv')
test_volume = pd.read_csv(FEATUREPATH+'time_volume_features_test.csv')

print train_stat.shape
print train_volume.shape
print test_stat.shape
print test_volume.shape

link_feat = pd.read_csv(FEATUREPATH+'links_features.csv')
link_volume = pd.read_csv(FEATUREPATH+'linkVolume.csv')
link_time = pd.read_csv(FEATUREPATH+'linkTime.csv')
link_speed = pd.read_csv(FEATUREPATH+'linkSpeed.csv')
weather = pd.read_csv(FEATUREPATH+'weatherFeat.csv')

def merge_features(version = 0):
	if version == 0:
		train =pd.merge(train_stat, train_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_feat,on=['tollgate_id','intersection_id'],how='left')
		print train.shape,train.dropna().shape

		train = pd.merge(train,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train, link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,weather,on=['date','hour'],how='left')
		print train.shape,train.dropna().shape

		test =pd.merge(test_stat,test_volume,on=['tollgate_id','intersection_id','hour','date'],how='left')
		test = pd.merge(test,link_feat,on=['tollgate_id','intersection_id'],how='left')
		test = pd.merge(test,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')

		test = pd.merge(test,weather,on=['date','hour'],how='left')
		print test.shape,test.dropna().shape


		train['is_weekend'] = 0
		train.loc[train.weekday.isin([5,6]),'is_weekend'] = 1

		test['is_weekend'] = 0
		test.loc[test.weekday.isin([5,6]),'is_weekend'] = 1

		#time_zone split
		#train['time_zone'] = 0
		#train.loc[train.hour.isin([8,9,1]),'time_zone'] = 1
		#train.loc[train.hour.isin([11,12,13]),'time_zone'] = 2
		#train.loc[train.hour.isin([14,15,16]),'time_zone'] = 3
		#train.loc[train.hour.isin([17,18,19]),'time_zone'] = 4


		time_zone = [7,11,13,16,18,23]
		zone = range(5)
		train['time_zone'] = pd.cut(train.hour,bins=time_zone,labels=zone).astype(int)
		test['time_zone'] = pd.cut(test.hour,bins=time_zone,labels=zone).astype(int)


		train['time_zone2'] = train['time_zone']*10+train['is_weekend']
		test['time_zone2'] = test['time_zone']*10+test['is_weekend']
		train['is_holiday'] = 0
		train.loc[train.date.isin(range(1001,1008)),'is_holiday'] = 1
		train.loc[train.date.isin(range(915,918)),'is_holiday'] = 1
		test['is_holiday'] = 0

		test['time_zone'] = 0
		test.loc[test.hour.isin([8,9,10]),'time_zone'] = 1
		test.loc[test.hour.isin([17,18,19]),'time_zone'] = 4
		#road split
		train['road'] = 0
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==1)),'road'] = 1
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==2)),'road'] = 2
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==3)),'road'] = 3
		train.loc[((train.intersection_id=='B')&(train.tollgate_id==3)),'road'] = 4
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==3)),'road'] = 5

		test['road'] = 0
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==1)),'road'] = 1
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==2)),'road'] = 2
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==3)),'road'] = 3
		test.loc[((test.intersection_id=='B')&(test.tollgate_id==3)),'road'] = 4
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==3)),'road'] = 5

	if version == 1:
		train =pd.merge(train_stat, train_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_feat,on=['tollgate_id','intersection_id'],how='left')
		print train.shape,train.dropna().shape

		train = pd.merge(train,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		# train = pd.merge(train, link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,weather,on=['date','hour'],how='left')
		print train.shape,train.dropna().shape

		test =pd.merge(test_stat,test_volume,on=['tollgate_id','intersection_id','hour','date'],how='left')
		test = pd.merge(test,link_feat,on=['tollgate_id','intersection_id'],how='left')
		test = pd.merge(test,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		# test = pd.merge(test,link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')

		test = pd.merge(test,weather,on=['date','hour'],how='left')
		print test.shape,test.dropna().shape


		train['is_weekend'] = 0
		train.loc[train.weekday.isin([5,6]),'is_weekend'] = 1

		test['is_weekend'] = 0
		test.loc[test.weekday.isin([5,6]),'is_weekend'] = 1

		time_zone = [7,11,13,16,18,23]
		zone = range(5)
		train['time_zone'] = pd.cut(train.hour,bins=time_zone,labels=zone).astype(int)
		test['time_zone'] = pd.cut(test.hour,bins=time_zone,labels=zone).astype(int)


		train['time_zone2'] = train['time_zone']*10+train['is_weekend']
		test['time_zone2'] = test['time_zone']*10+test['is_weekend']
		train['is_holiday'] = 0
		train.loc[train.date.isin(range(1001,1008)),'is_holiday'] = 1
		train.loc[train.date.isin(range(915,918)),'is_holiday'] = 1
		test['is_holiday'] = 0

		test['time_zone'] = 0
		test.loc[test.hour.isin([8,9,10]),'time_zone'] = 1
		test.loc[test.hour.isin([17,18,19]),'time_zone'] = 4

		train['road'] = 0
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==1)),'road'] = 1
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==2)),'road'] = 2
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==3)),'road'] = 3
		train.loc[((train.intersection_id=='B')&(train.tollgate_id==3)),'road'] = 4
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==3)),'road'] = 5

		test['road'] = 0
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==1)),'road'] = 1
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==2)),'road'] = 2
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==3)),'road'] = 3
		test.loc[((test.intersection_id=='B')&(test.tollgate_id==3)),'road'] = 4
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==3)),'road'] = 5

	if version == 2:
		train =pd.merge(train_stat, train_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_feat,on=['tollgate_id','intersection_id'],how='left')
		print train.shape,train.dropna().shape

		train = pd.merge(train,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		# train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train, link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,weather,on=['date','hour'],how='left')
		print train.shape,train.dropna().shape

		test =pd.merge(test_stat,test_volume,on=['tollgate_id','intersection_id','hour','date'],how='left')
		test = pd.merge(test,link_feat,on=['tollgate_id','intersection_id'],how='left')
		test = pd.merge(test,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		# test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')

		test = pd.merge(test,weather,on=['date','hour'],how='left')
		print test.shape,test.dropna().shape



		train['is_weekend'] = 0
		train.loc[train.weekday.isin([5,6]),'is_weekend'] = 1

		test['is_weekend'] = 0
		test.loc[test.weekday.isin([5,6]),'is_weekend'] = 1

		#time_zone split
		#train['time_zone'] = 0
		#train.loc[train.hour.isin([8,9,1]),'time_zone'] = 1
		#train.loc[train.hour.isin([11,12,13]),'time_zone'] = 2
		#train.loc[train.hour.isin([14,15,16]),'time_zone'] = 3
		#train.loc[train.hour.isin([17,18,19]),'time_zone'] = 4


		time_zone = [7,11,13,16,18,23]
		zone = range(5)
		train['time_zone'] = pd.cut(train.hour,bins=time_zone,labels=zone).astype(int)
		test['time_zone'] = pd.cut(test.hour,bins=time_zone,labels=zone).astype(int)


		train['time_zone2'] = train['time_zone']*10+train['is_weekend']
		test['time_zone2'] = test['time_zone']*10+test['is_weekend']
		train['is_holiday'] = 0
		train.loc[train.date.isin(range(1001,1008)),'is_holiday'] = 1
		train.loc[train.date.isin(range(915,918)),'is_holiday'] = 1
		test['is_holiday'] = 0

		test['time_zone'] = 0
		test.loc[test.hour.isin([8,9,10]),'time_zone'] = 1
		test.loc[test.hour.isin([17,18,19]),'time_zone'] = 4
		#road split
		train['road'] = 0
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==1)),'road'] = 1
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==2)),'road'] = 2
		train.loc[((train.intersection_id=='A')&(train.tollgate_id==3)),'road'] = 3
		train.loc[((train.intersection_id=='B')&(train.tollgate_id==3)),'road'] = 4
		train.loc[((train.intersection_id=='C')&(train.tollgate_id==3)),'road'] = 5

		test['road'] = 0
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==1)),'road'] = 1
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==2)),'road'] = 2
		test.loc[((test.intersection_id=='A')&(test.tollgate_id==3)),'road'] = 3
		test.loc[((test.intersection_id=='B')&(test.tollgate_id==3)),'road'] = 4
		test.loc[((test.intersection_id=='C')&(test.tollgate_id==3)),'road'] = 5

	if version == 3:
		train =pd.merge(train_stat, train_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_feat,on=['tollgate_id','intersection_id'],how='left')
		print train.shape,train.dropna().shape

		train = pd.merge(train,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train, link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')
		train = pd.merge(train,weather,on=['date','hour'],how='left')
		print train.shape,train.dropna().shape

		test =pd.merge(test_stat,test_volume,on=['tollgate_id','intersection_id','hour','date'],how='left')
		test = pd.merge(test,link_feat,on=['tollgate_id','intersection_id'],how='left')
		test = pd.merge(test,link_volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_time,on=['tollgate_id','intersection_id','date','hour'],how='left')
		test = pd.merge(test,link_speed,on=['tollgate_id','intersection_id','date','hour'],how='left')

		test = pd.merge(test,weather,on=['date','hour'],how='left')
		print test.shape,test.dropna().shape


	return train, test

def evalmape(preds,dtrain):
    gaps = dtrain.get_label()
    err = abs(gaps-preds)/gaps
    err[(gaps==0)] = 10**-10
    err = np.mean(err)
    return 'MAPE',err

def train_model():
	weight = 1 / np.log(trainLabel)
	dtrain = xgb.DMatrix(trainFeature, trainLabel, weight=weight)
	dtest = xgb.DMatrix(testFeature)
	import random
	NFOLDS = 20
	ntest = testFeature.shape[0]
	test_skf = np.empty((NFOLDS, ntest))

	for i in range(NFOLDS):
	    print 'training '+str(i)
	    params={
	    #     'scale_pos_weight': 1,
	        'booster':'gbtree',
	        'objective': 'reg:linear',
	        'eval_metric': 'mae',
	        'stratified':True,
	        'max_depth':random.randint(7, 9),
	        'gamma':0.1,
	        'min_child_weight':random.randint(1, 2),
	        'subsample':random.uniform(0.65, 0.85),
	        'colsample_bytree':random.uniform(0.65, 0.85),

	    #     'lambda':0.001,   #550
	        'alpha':0.00001,
	    #     'lambda_bias':0.1,

	        'eta': 0.01,
	        'seed':random.randint(0, 1024*1024),    
	        'silent':1
	    }


	    rounds = random.randint(280, 320)
	    folds = 5
	    num_round = rounds
	    watchlist = [(dtrain, 'train')]
	    model = xgb.train(params, dtrain, num_round, watchlist, feval=evalmape, early_stopping_rounds=100, verbose_eval=50)
	    test_skf[i, :] = model.predict(dtest)

	return test_skf.mean(axis=0)


train, test = merge_features(version=0)
trainFeature = train.drop(['label','date','intersection_id'],axis=1)
trainLabel = train['label']
testFeature = test.drop('time_window',axis=1)
testIndex = test[['intersection_id','tollgate_id','time_window']]
testFeature = testFeature[trainFeature.columns]
print trainFeature.shape
print testFeature.shape
print 'creating preds0...'
preds0 = train_model()

train, test = merge_features(version=1)
trainFeature = train.drop(['label','date','intersection_id'],axis=1)
trainLabel = train['label']
testFeature = test.drop('time_window',axis=1)
testIndex = test[['intersection_id','tollgate_id','time_window']]
testFeature = testFeature[trainFeature.columns]
print trainFeature.shape
print testFeature.shape
print 'creating preds1...'
preds1 = train_model()

train, test = merge_features(version=2)
trainFeature = train.drop(['label','date','intersection_id'],axis=1)
trainLabel = train['label']
testFeature = test.drop('time_window',axis=1)
testIndex = test[['intersection_id','tollgate_id','time_window']]
testFeature = testFeature[trainFeature.columns]
print trainFeature.shape
print testFeature.shape
print 'creating preds2...'
preds2 = train_model()

train, test = merge_features(version=3)
trainFeature = train.drop(['label','date','intersection_id'],axis=1)
trainLabel = train['label']
testFeature = test.drop('time_window',axis=1)
testIndex = test[['intersection_id','tollgate_id','time_window']]
testFeature = testFeature[trainFeature.columns]
print trainFeature.shape
print testFeature.shape
print 'creating preds3...'
preds3 = train_model()

out_df = test[['intersection_id','tollgate_id','time_window']]
out_df['avg_travel_time'] = preds0*0.4+preds1*0.2+preds2*0.2+preds3*0.2
out_df.to_csv(RESULTPATH+RESULTNAME, index=0)