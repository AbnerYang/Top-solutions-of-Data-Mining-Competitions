# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
import time
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, KFold
from model import *

def LogInfo(stri):
    print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + ' ' + stri

# In[44]:


LOCAL_TRAIN_PATH = "../data/local_train.csv"
LOCAL_TEST_PATH = "../data/local_test.csv"

ONLINE_TEST_PATH = "../data/final_test_use.csv"


# ['candidate_shop_id', 'row_id', 'weighted_membership_shopvector_cumprod', 'weighted_membership_shopvector_cumsum', 'weighted_membership_shopvector_orMax', 'based_prob_shopvector_cumprod', 'based_prob_shopvector_cumsum', 'based_prob_shopvector_orMax', 'weighted_prob_shopvector_cumprod', 'weighted_prob_shopvector_cumsum', 'weighted_prob_shopvector_orMax', 'user_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos', 'shop_category_id', 'shop_longitude', 'shop_latitude', 'shop_price', 'shop_mall_id', 'day', 'moment', 'cmoment', 'wday', 'label']
# In[46]:

drop_cols =  ['candidate_shop_id', 'row_id', 'user_id', 'time_stamp', 'longitude', 'latitude', 'wifi_infos', 'shop_longitude', 'shop_latitude', 'shop_price', 'cmoment', 'day']

str_to_int_cols = ['shop_mall_id', 'shop_category_id']
feature_cols = ['weighted_membership_shopvector_cumprod', 'weighted_membership_shopvector_cumsum', 'weighted_membership_shopvector_orMax', 'based_prob_shopvector_cumprod', 'based_prob_shopvector_cumsum', 'based_prob_shopvector_orMax', 'weighted_prob_shopvector_cumprod', 'weighted_prob_shopvector_cumsum', 'weighted_prob_shopvector_orMax', 'shop_mall_id', 'shop_category_id', 'wday', 'moment']
final_use_cols = ['candidate_shop_id', 'row_id', 'label']
# df_train_index = df_online_train[['row_id', 'user_id']]

### local train_test
xgb_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.1,

    'lambda': 0.1,
    'scale_pos_weight': 1,
    'gamma': 0.65,
    'min_child_weight': 10,

    'nthread': 18,
    'silent': 1
}
SEED = 1126

##############
#   local    #
##############
'''
df_train = pd.read_csv(LOCAL_TRAIN_PATH)
df_test = pd.read_csv(LOCAL_TEST_PATH)
df_train_remain = df_train[final_use_cols]
df_test_remain = df_test[final_use_cols]

df_train = df_train.drop(drop_cols, axis=1)
df_test = df_test.drop(drop_cols, axis=1)

process_list = [df_train, df_test]
for i in range(len(process_list)):
    for col in str_to_int_cols:
        process_list[i][col] = process_list[i].apply(lambda row: int(row[col][2:]), axis = 1)

print df_train.shape, df_test.shape
print df_train.head(), df_test.head()

df_train_y = df_train[['label']].values
df_train_x = df_train.drop('label', axis=1).values

df_test_y = df_test[['label']].values
df_test_x = df_test.drop('label', axis=1).values

xgbModel = XgbWrapper(seed=SEED, params=xgb_params, rounds=70, has_weight=False)
frame_online = MLframe(df_train_x, df_train_y, df_test_x)
#frame_online.train_test(xgbModel, df_test_y, isxgb = True)
predict = frame_online.train_predict(xgbModel)

# In[102]:
res = df_test_remain
res['prob'] = predict
res['rank'] = res.groupby('row_id')['prob'].rank(method='min', ascending=False)
res = res[res['rank'] < 2]
#print res.head()
#res = res.groupby('row_id')['shop_id'].max().reset_index()
print 'acc', float(res.label.sum()) / res.label.count()
res[final_use_cols].to_csv('../data/local_pred_top10.csv', index=None)
'''

##############
#   online   #
##############
df_train = pd.read_csv(LOCAL_TRAIN_PATH)
df_test = pd.read_csv(LOCAL_TEST_PATH)
df_online = pd.read_csv(ONLINE_TEST_PATH)

df_train_remain = df_train[final_use_cols]
df_test_remain = df_test[final_use_cols]
df_online_remain = df_online[['candidate_shop_id', 'row_id']]

process_list = [df_train, df_test, df_online]
for i in range(len(process_list)):
    process_list[i] = process_list[i].drop(drop_cols, axis = 1)
    for col in str_to_int_cols:
        process_list[i][col] = process_list[i].apply(lambda row: int(row[col][2:]), axis = 1)

print df_train.shape, df_test.shape, df_online.shape
print df_train.head(), df_test.head(), df_online.head()

df_train = pd.concat([df_train, df_test], axis = 0)

df_train_y = df_train[['label']].values
df_train_x = df_train.drop('label', axis=1).values

xgbModel = XgbWrapper(seed=SEED, params=xgb_params, rounds=100, has_weight=False)
frame_online = MLframe(df_train_x, df_train_y, df_online)
#frame_online.train_test(xgbModel, df_test_y, isxgb = True)
predict = frame_online.train_predict(xgbModel)

# In[102]:
res = df_online_remain
res['prob'] = predict
res['rank'] = res.groupby('row_id')['prob'].rank(method='min', ascending=False)
res = res[res['rank'] < 2]
#print res.head()
#res = res.groupby('row_id')['shop_id'].max().reset_index()
#print 'acc', float(res.label.sum()) / res.label.count()
res[final_use_cols].to_csv('../data/online_pred.csv', index=None)
