import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
import numpy as np
import pandas as pd

FEATURES_PATH = '../../feature/xgb-1/'
RESULT_PATH =  '../../result/'
RESULT_NAME = 'xgb-1.csv'

def xgbCVModel(trainFeature, trainLabel, rounds, folds, params):
    weight = 1 / np.log(trainLabel)
    dtrain = xgb.DMatrix(trainFeature, label=trainLabel, weight=weight)
    num_rounds = rounds
    print 'run CV: rounds:', rounds, ' folds: ', folds
    res = xgb.cv(params, dtrain, num_rounds, nfold=folds, verbose_eval=10, feval=evalmape, early_stopping_rounds=50)
    return res


def xgbLocalModel(trainFeature, testFeature, trainLabel, testLabel, params, rounds, is_weight=True):
    if is_weight:
        weight = 1 / np.log(trainLabel)
        dtrain = xgb.DMatrix(trainFeature, label=trainLabel, weight=weight)
    else:
        dtrain = xgb.DMatrix(trainFeature, label=trainLabel)
    dtest = xgb.DMatrix(testFeature, label=testLabel)
    watchlist = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = rounds
    print 'run local: ' + 'round: ' + str(rounds)
    model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=50, feval=evalmape, early_stopping_rounds=50)
    predict = model.predict(dtest)
    evalm = mape(testLabel, predict)
    print evalm
    return model.best_iteration


def xgbPredict(trainFeature, trainLabel, testFeature, rounds, params, is_weight=True):
    if is_weight:
        weight = 1 / np.log(trainLabel)
        dtrain = xgb.DMatrix(trainFeature, label=trainLabel, weight=weight)
    else:
        dtrain = xgb.DMatrix(trainFeature, label=trainLabel)
    dtrain = xgb.DMatrix(trainFeature, label=trainLabel)
    dtest = xgb.DMatrix(testFeature, label=np.zeros(testFeature.shape[0]))
    watchlist = [(dtrain, 'train')]
    num_round = rounds
    model = xgb.train(params, dtrain, num_round, watchlist, verbose_eval=50, feval=evalmape)
    predict = model.predict(dtest)
    return model, predict


def evalmape(preds, dtrain):
    gaps = dtrain.get_label()
    err = abs(gaps - preds) / gaps
    err[(gaps == 0)] = 10 ** -10
    err = np.mean(err)
    return 'MAPE', err

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


# Load features
print 'Loading features...'
train = pd.read_csv(FEATURES_PATH + 'trainFeature_stat.csv')
test = pd.read_csv(FEATURES_PATH + 'testFeature_stat.csv')
links = pd.read_csv(FEATURES_PATH + 'links_features.csv')
volume = pd.read_csv(FEATURES_PATH + 'time_volume_features_train.csv')
volume_test = pd.read_csv(FEATURES_PATH + 'time_volume_features_test.csv')
linkVolume = pd.read_csv(FEATURES_PATH + 'linkVolume.csv')
linkTime = pd.read_csv(FEATURES_PATH + 'linkTime.csv')
weather = pd.read_csv(FEATURES_PATH + 'weatherFeat.csv')

# Merge features
print 'Merging features...'
train =pd.merge(train,volume,on=['tollgate_id','intersection_id','date','hour'],how='left')
train = pd.merge(train,links,on=['tollgate_id','intersection_id'],how='left')
train = pd.merge(train,weather,on=['date','hour'],how='left')
train = pd.merge(train,linkVolume,on=['tollgate_id','intersection_id','date','hour'],how='left')
train = pd.merge(train,linkTime,on=['tollgate_id','intersection_id','date','hour'],how='left')
train = pd.merge(train,linkTime,on=['tollgate_id','intersection_id','date','hour'],how='left')
test =pd.merge(test,volume_test,on=['tollgate_id','intersection_id','hour','date'],how='left')
test = pd.merge(test,links,on=['tollgate_id','intersection_id'],how='left')
test = pd.merge(test,linkVolume,on=['tollgate_id','intersection_id','date','hour'],how='left')
test = pd.merge(test,linkTime,on=['tollgate_id','intersection_id','date','hour'],how='left')
test = pd.merge(test,linkTime,on=['tollgate_id','intersection_id','date','hour'],how='left')
test = pd.merge(test,weather,on=['date','hour'],how='left')

### Add index feature
print 'Adding index feature...'
# about time
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

#about route
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

print train.shape, test.shape


trainSet = train[(train.date<1011)]

###offline train-test to find the best iter-rounds
timelst = [8,17]
testSet = train[(train.hour.isin(timelst))&(train.date>=1011)&(train.date<1018)]
trainFeature = trainSet.drop(['label','date','intersection_id'],axis=1)
trainLabel = trainSet['label']
testLabel = testSet.label.values
testFeature = testSet.drop(['label','date','intersection_id'],axis=1)
testFeature = testFeature[trainFeature.columns]
testIdx = testSet[['date','hour']]
print testFeature.shape
print trainFeature.shape

config = {
    'rounds': 10000,
    'folds': 5
}

params = {
    'booster': 'gbtree',
    'objective': 'reg:linear',
    'stratified': True,
    'max_depth': 8,
    'min_child_weight': 1,
    'gamma': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda': 1,
    'eta': 0.01,
    'seed': 20,
    'silent': 1
}

# Use both the train-val and cross validation to decide the best rounds
# bst_rounds = xgbLocalModel(trainFeature,testFeature,trainLabel,testLabel,params,config['rounds'],is_weight=True)
# res = xgbCVModel(trainFeature,trainLabel,config['rounds'],config['folds'],params)

# Find the best rounds
# bst_rounds1 = len(res)
mround = 308
print 'best rounds is ',308

trainFeature = train.drop(['label','date','intersection_id'],axis=1)
trainLabel = train['label']
testFeature = test.drop('time_window',axis=1)
testIndex = test[['intersection_id','tollgate_id','time_window']]
testFeature = testFeature[trainFeature.columns]

model,predict = xgbPredict(trainFeature,trainLabel,testFeature,mround,params,is_weight=True)
#result = storeResult(testIndex,predict,RESULT_NAME)
result = testIndex
result['avg_travel_time'] = predict
print np.mean(predict)
result.to_csv(RESULT_PATH + RESULT_NAME,index=0)
