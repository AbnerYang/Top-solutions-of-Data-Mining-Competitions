# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:32:24 2017

@author: 1
"""

import pandas as pd
import numpy as np
import time
import sys
import pickle
import os
import collections
import copy
import math

#%% initialization
has_generate_prefeature_file = False
is_online = 'no'
#%% tools code by karon
def log(info):
    print time.strftime("[%Y-%m-%d %H:%M:%S]", time.gmtime()), info

def calc_distances(xyxy):
    x1, y1, x2, y2 = xyxy
    x1, y1, x2, y2 = map(lambda s: s.astype(float), [x1, y1, x2, y2])
    x1, y1, x2, y2 = map(np.radians, [x1, y1, x2, y2])
    dx = x2 - x1
    dy = y2 - y1
    a = np.sin(dy / 2) ** 2 + np.cos(y1) * np.cos(y2) * np.power(np.sin(dx / 2), 2)
    c = 2 * np.sqrt(a).map(lambda x: math.asin(x))
    r = 6371
    return c * r * 1000

def calc_distance(x1, y1, x2, y2):
    x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
    x1, y1, x2, y2 = map(math.radians, [x1, y1, x2, y2])
    dx = x2 - x1
    dy = y2 - y1
    a = math.sin(dy / 2) ** 2 + math.cos(y1) * math.cos(y2) * math.sin(dx / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r * 1000

def read(path):
    df = pd.read_csv(path)
    return df
# 把wifi强度变成越大越好的非负值
def inverse_force(wifi_force):
    return np.max([0, int(wifi_force) + 100])

def transfer_super_small_num(val):
    return 1.0 / (1.0 + np.sqrt(np.abs(np.log(val))))
#%% preprocess code by karon
def read_all():
    shop_info = read(u'../data/训练数据-ccf_first_round_shop_info.csv')
    user_log = read(u'../data/训练数据-ccf_first_round_user_shop_behavior.csv')
    evaluation = read(u'../data/AB榜测试集-evaluation_public.csv')
    return shop_info, user_log, evaluation

def get_day_moment(df, bias):
    cdf = df.copy()
    tmp = df.time_stamp.map(lambda t: time.strptime(t, u'%Y-%m-%d %H:%M'))
    cdf['day'] = tmp.map(lambda t: t.tm_mday + bias)
    cdf['moment'] = tmp.map(lambda t: t.tm_hour * 6 + t.tm_min / 10)
    cdf['cmoment'] = 24 * 6 * cdf['day'] + cdf['moment'] - 144
    cdf['wday'] = tmp.map(lambda t: t.tm_wday + 1)
    return cdf

def preprocess():
    shop_info, user_log, evaluation = read_all()
    shop_info = shop_info.add_prefix("shop_")
    shop_info.rename(columns = {'shop_shop_id': 'shop_id'}, inplace = True)
    user_shop = pd.merge(user_log, shop_info, on = "shop_id", how = "left")
    # transfer time_stamp into more user field
    user_shop = get_day_moment(user_shop, 0)
    evaluation = get_day_moment(evaluation, 31)
    user_shop.to_csv('../data/0801-0831.csv', index=None) 
    user_shop[user_shop.time_stamp < "2017-08-17 00:00"].to_csv("../data/0801-0817.csv", index = None)
    user_shop[user_shop.time_stamp >= "2017-08-17 00:00"].to_csv("../data/0817-0831.csv", index = None)
    evaluation = evaluation.rename(columns={'mall_id': 'shop_mall_id'})
    evaluation.to_csv("../data/new_evaluation_public.csv", index = None)



# preprocess
if not os.path.exists('../data/0817-0831.csv') or not has_generate_prefeature_file:
    log('preprocess')
    preprocess()
#%% define train and test set
is_offline = (is_online != 'yes')

data_sets = {}
if not has_generate_prefeature_file:
    log('reload train and test set')
    shop_info = read(u'../data/训练数据-ccf_first_round_shop_info.csv')
    groups = shop_info.groupby('mall_id')
    mall_shoplist = {}
    for mall_id, gp in groups:
        mall_shoplist[mall_id] = gp.shop_id.values
        
    if is_offline:
        log("loading %s set"  % ('offline' if is_offline else 'online'))
        data_sets = dict(train=pd.read_csv('../data/0801-0817.csv'), 
                          test=pd.read_csv('../data/0817-0831.csv'))
    else:
        log("loading %s set"  % ('offline' if is_offline else 'online'))
        data_sets =  dict(train=pd.read_csv('../data/0801-0831.csv'), 
                          test=pd.read_csv(u'../data/new_evaluation_public.csv'))
#%% clean wifi (also can choose append wifi) 
# get wifi in train and test set at same time
def get_wifi_list(df):
    tmp = str.join(';', df.wifi_infos.tolist())
    eleL = tmp.split(';')
    wifi_id = [x.split('|')[0] for x in eleL]
    return eleL, wifi_id
def get_wifiids(wifi_infos):
    eleL = wifi_infos.split(';')
    wifi_id = [x.split('|')[0] for x in eleL]
    return set(wifi_id)


if not has_generate_prefeature_file:
    log('get mall wifi id list')
    tmp = data_sets['train'].groupby('shop_mall_id', 
                                       as_index=False).apply(lambda g: [g.shop_mall_id.iloc[0], 
                                                             str.join(';', g.wifi_infos.tolist())])
    mall_wifilist = {key: get_wifiids(val) for key, val in tmp.tolist()}
        
    log('get joint wifi id list')
    wifi_ids = {}
    for set_key, data_set in data_sets.items():
        _, wifi_ids[set_key] = get_wifi_list(data_set)
    
    joint_wifi_id = wifi_ids['train'] + wifi_ids['test']
    tmp = pd.Series(joint_wifi_id).value_counts().reset_index()
    tmp.columns = ['wifi_id', 'counts']
    joint_wifi_id = set(tmp.loc[(tmp.counts > 5)].wifi_id)
print(len(joint_wifi_id))
# wifi_filter process
def clean_wifi_infos(df, wifi_ids):
    cdf = df.copy()
    def wifi_filter(wifis):
        clean = []
        tmp = wifis.split(';')
        for x in tmp:
            if x.split('|')[0] in joint_wifi_id:
                clean.append(x)
        return str.join(';', clean)
    cdf['wifi_infos'] = cdf.wifi_infos.map(wifi_filter)
    cover_rec_rate = len(cdf.loc[cdf.wifi_infos != '', :]) / float(len(cdf))
    return cdf, cover_rec_rate


if not os.path.exists('../data/prefeature_test_%s.csv' % ('offline' if is_offline else 'online')) or not has_generate_prefeature_file:
    log('clean wifi infos')
    for set_key, data_set in data_sets.items():
        data_sets[set_key], _ = clean_wifi_infos(data_set, joint_wifi_id)
        L = len(data_sets[set_key])
        data_sets[set_key]['row_id'] = map(str, np.arange(L))
        #data_sets[set_key].to_csv('../data/prefeature_%s_%s.csv' % (set_key, ('offline' if is_offline else 'online')))
        print _
#%% --------------------feature construct--------------------

if has_generate_prefeature_file:
    data_sets = {}
    for set_key in ['train', 'test']:
        data_sets[set_key] = pd.read_csv('../data/prefeature_%s_%s.csv' % (set_key, ('offline' if is_offline else 'online')))
tr_set = data_sets['train']
te_set = data_sets['test']

#%% 用户特征
## 训练集用户到过店铺的次数， 用户的记录数
user_shopnum = tr_set.groupby(['user_id', 'shop_id'], as_index=False)[['shop_mall_id']].count()
user_shopnum.rename(columns={'shop_mall_id': 'user_shop_num'}, inplace=True)

user_recnum = tr_set.groupby(['user_id'], as_index=False)[['shop_mall_id']].count()
user_recnum.rename(columns={'shop_mall_id': 'user_rec_num'}, inplace=True)

## 训练集用户到过category的次数
user_categorynum = tr_set.groupby(['user_id', 'shop_category_id'], as_index=False)[['shop_mall_id']].count()
user_categorynum.rename(columns={'shop_mall_id': 'user_category_num'}, inplace=True)

#%% 记录特征 (用到未来数据)
## 前后60分钟内的多长时间有新的消费行为(用自身数据集)
tmp_user_recnum = te_set.groupby(['user_id'], as_index=False)[['shop_mall_id']].count()
tmp_user_recnum.rename(columns={'shop_mall_id': 'te_user_rec_num'}, inplace=True)
tmp_user_recnum = tmp_user_recnum.loc[tmp_user_recnum.te_user_rec_num == 2, :]
pte_set = tmp_user_recnum[['user_id']].merge(te_set, on='user_id', how='left').sort_values(['user_id', 'time_stamp'])
diff1 = pte_set.cmoment.diff()
diff2 = pte_set.cmoment.diff(-1)
L = len(diff1)
diff = pd.concat([diff1.iloc[range(1, L, 2)], diff2.iloc[range(0, L, 2)]], axis=0).map(np.abs)
pte_set['surround_rec_diff'] = diff.map(lambda x: -1 if x > 6 else x)

te_set = te_set.merge(pte_set.loc[:,['row_id', 'surround_rec_diff']], how='left', on='row_id')
te_set.surround_rec_diff.fillna(-1, inplace=True)

#%% 店铺特征
## 店铺的所有记录数，同一用户半小时内进行两次消费的记录数，比例
shop_info = pd.read_csv(u'../data/训练数据-ccf_first_round_shop_info.csv')
shop_info = shop_info.add_prefix("shop_")
shop_info.rename(columns={'shop_shop_id': 'shop_id'}, inplace=True)

tmp_user_recnum = tr_set.groupby(['user_id'], as_index=False)[['shop_mall_id']].count()
tmp_user_recnum.rename(columns={'shop_mall_id': 'tr_user_rec_num'}, inplace=True)
tmp_user_recnum = tmp_user_recnum.loc[tmp_user_recnum.tr_user_rec_num == 2, :]
ptr_set = tmp_user_recnum[['user_id']].merge(tr_set, on='user_id', how='left').sort_values(['user_id', 'time_stamp'])
diff1 = ptr_set.cmoment.diff()
diff2 = ptr_set.cmoment.diff(-1)
L = len(diff1)
diff = pd.concat([diff1.iloc[range(1, L, 2)], diff2.iloc[range(0, L, 2)]], axis=0).map(np.abs)
ptr_set['surround_rec_diff'] = diff.map(lambda x: -1 if x > 3 else x)
ptr_set = ptr_set.loc[ptr_set.surround_rec_diff >= 0, :]

contRec_num = ptr_set.shop_id.value_counts().reset_index()
contRec_num.columns = ['shop_id', 'contRec_num']
shop_info = shop_info.merge(contRec_num, on='shop_id', how='left')

totalRec_num = tr_set.groupby('shop_id', as_index=False)[['row_id']].count()
totalRec_num.rename(columns={'row_id': 'totalRec_num'}, inplace=True)
shop_info = shop_info.merge(totalRec_num, on='shop_id', how='left')

shop_info['contRec_rate'] = shop_info['contRec_num'] / shop_info['totalRec_num'].astype(np.float)

## 店铺的记录八个方向的定位误差统计量（均值、方差、数量、比例）
shop_cord = tr_set.loc[:, ['shop_id', 'shop_mall_id', 'longitude', 'shop_longitude', 'latitude', 'shop_latitude']].copy()
shop_cord['dis_longitude'] = shop_cord['longitude'] - shop_cord['shop_longitude'] + 1e-6
shop_cord['dis_latitude'] = shop_cord['latitude'] - shop_cord['shop_latitude'] + 1e-6
shop_cord['radians'] = (shop_cord.dis_latitude / shop_cord.dis_longitude).map(lambda x: 0 if np.abs(x) >= 1 else 4)
shop_cord['side'] = shop_cord.dis_latitude.map(lambda x: 0 if x >= 0 else 2) + shop_cord.dis_longitude.map(lambda x: 0 if x >= 0 else 1)
shop_cord['direction'] = shop_cord['radians'] + shop_cord['side'] 
cord_array2d = shop_cord.loc[:, ['longitude', 'latitude', 'shop_longitude' ,'shop_latitude']].values

tr_set['direction'] = shop_cord['direction']
tr_set['distance'] = map(calc_distance, cord_array2d)
shopdisterror = tr_set.groupby(['shop_id', 'direction'])[['distance']].agg([np.median, np.std, np.count_nonzero]).fillna(0)
shopdisterror.columns = ['distance_median' , 'distance_std', 'direct_count']
shopdisterror = shopdisterror.reset_index()
shopdisterror.columns = ['shop_id', 'direction', 'distance_median' , 'distance_std', 'direct_count']

#%% 商场特征
tmp = {u'm_1021': 1, u'm_1085': 9, u'm_1089': 6, u'm_1175': 5, u'm_1263': 3, u'm_1293': 1, u'm_1375': 5, u'm_1377': 2, u'm_1409': 3, u'm_1621': 1, u'm_1790': 0, u'm_1831': 9, 
u'm_1920': 1, u'm_1950': 3, u'm_2009': 0, u'm_2058': 8, u'm_2123': 4, u'm_2182': 5, u'm_2224': 1, u'm_2267': 2, u'm_2270': 1, u'm_2333': 10, u'm_2415': 8, u'm_2467': 1, 
u'm_2578': 9, u'm_2715': 1, u'm_2878': 4, u'm_2907': 9, u'm_3005': 5, u'm_3019': 1, u'm_3054': 1, u'm_3112': 1, u'm_3313': 1, u'm_3425': 8, u'm_3445': 3, u'm_3501': 5, 
u'm_3517': 1, u'm_3528': 5, u'm_3739': 0, u'm_3832': 0, u'm_3839': 4, u'm_3871': 8, u'm_3916': 10, u'm_4011': 8, u'm_4033': 2, u'm_4079': 4, u'm_4094': 0, u'm_4121': 8, 
u'm_4168': 1, u'm_4187': 9, u'm_4341': 5, u'm_4406': 3, u'm_4422': 1, u'm_4459': 1, u'm_4495': 8, u'm_4515': 5, u'm_4543': 10, u'm_4548': 8, u'm_4572': 7, u'm_4759': 3, 
u'm_4828': 3, u'm_4923': 8, u'm_5076': 8, u'm_5085': 7, u'm_5154': 8, u'm_5352': 0, u'm_5529': 9, u'm_5767': 8, u'm_5810': 1, u'm_5825': 3, u'm_5892': 5, u'm_615': 7, 
u'm_6167': 2, u'm_622': 3, u'm_623': 10, u'm_625': 3, u'm_626': 5, u'm_6337': 1, u'm_6587': 5, u'm_6803': 2, u'm_690': 1, u'm_7168': 9, u'm_7374': 3, u'm_7523': 3, 
u'm_7601': 2, u'm_7800': 1, u'm_7973': 2, u'm_7994': 3, u'm_8093': 7, u'm_822': 8, u'm_8344': 3, u'm_8379': 8, u'm_9054': 1, u'm_9068': 7, u'm_909': 1, u'm_968': 3, u'm_979': 1}

mall_info = pd.DataFrame(dict(shop_mall_id=tmp.keys(), cluster=tmp.values()))

## 商场的店铺Price统计量、店铺数
price_statis = shop_info.groupby('shop_mall_id')[['shop_price']].agg([np.median, np.std, np.count_nonzero])
price_statis.columns = ['price_median', 'price_std', 'shop_count']
price_statis = price_statis.reset_index().rename(columns={'index': 'shop_mall_id'})
mall_info = mall_info.merge(price_statis, on='shop_mall_id', how='left')


set_key = 'train'
file_list = ['final_%s_%d_use' % (set_key, i+1) for i in range(20)]
cols = []
remain_cols = []
tr_cols = [u'shop_category_id', u'shop_longitude', u'shop_latitude',
        u'shop_price', u'shop_mall_id']
drop_cols = [u'time_stamp', u'wifi_infos', u'day', u'moment', u'cmoment', u'wday']

def reTopPercent(row_groups):
    L = len(row_groups) / 10 + 1
    topSample = row_groups.sort_values(by='based_prob_shopvector_logprod_no_mult_wifi_ratio', ascending=False).iloc[:L]
    return topSample

for idx, file_name in enumerate(file_list):
    unstack_final = pd.read_csv('../data/%s.csv' % file_name)
    if idx == 0:
        cols = unstack_final.columns
    unstack_final.columns = cols
    remain_cols = set(unstack_final.columns) - set(tr_cols) - set(drop_cols)
    unstack_final = unstack_final.loc[:, remain_cols].copy()
    
#    unstack_final = pd.concat(unstack_final.groupby('row_id').apply(reTopPercent).tolist(), axis=0)
#    
#    print unstack_final.columns
#    time.sleep(30)
    
    shop_info.rename(columns={'shop_id': u'candidate_shop_id'}, inplace=True)
    print shop_info.columns
    #time.sleep(10)
    unstack_final = pd.merge(unstack_final, shop_info, on='candidate_shop_id', how='left')

    #print(len(unstack_final))
    user_shopnum.rename(columns={'shop_id': 'candidate_shop_id'}, inplace=True)
    unstack_final = pd.merge(unstack_final, user_shopnum, on=['user_id', 'candidate_shop_id'], how='left')
    
    unstack_final = pd.merge(unstack_final, user_recnum, how='left', left_on='user_id',
                            right_on='user_id')
    unstack_final = pd.merge(unstack_final, user_categorynum, how='left', on=['user_id', 'shop_category_id'])

    
    unstack_final = pd.merge(unstack_final, te_set.loc[:, ['row_id', 'surround_rec_diff']], on='row_id',
                                        how='left')
    
    shop_cord = unstack_final.loc[:, ['shop_id', 'shop_mall_id', 'longitude', 'shop_longitude', 'latitude', 'shop_latitude']].copy()
    shop_cord['dis_longitude'] = shop_cord['longitude'] - shop_cord['shop_longitude'] + 1e-6
    shop_cord['dis_latitude'] = shop_cord['latitude'] - shop_cord['shop_latitude'] + 1e-6
    shop_cord['radians'] = (shop_cord.dis_latitude / shop_cord.dis_longitude).map(lambda x: 0 if np.abs(x) >= 1 else 4)
    shop_cord['side'] = shop_cord.dis_latitude.map(lambda x: 0 if x >= 0 else 2) + shop_cord.dis_longitude.map(lambda x: 0 if x >= 0 else 1)
    shop_cord['direction'] = shop_cord['radians'] + shop_cord['side'] 
    cord_array2d = shop_cord.loc[:, ['longitude', 'latitude', 'shop_longitude' ,'shop_latitude']].values
    
    unstack_final['direction'] = shop_cord['direction']
    unstack_final['distance'] = map(calc_distance, cord_array2d)    
    shopdisterror.rename(columns={'shop_id': 'candidate_shop_id'}, inplace=True)
    unstack_final = pd.merge(unstack_final, shopdisterror,
                                        on=['candidate_shop_id', 'direction'], how='left')
    
    unstack_final = pd.merge(unstack_final, mall_info, on='shop_mall_id', how='left')

    #print(len(unstack_final))
    unstack_final.drop(['user_id', 'longitude', 'latitude', 'shop_longitude', 'shop_latitude'], axis=1, inplace=True)
    unstack_final = unstack_final.fillna(0)
    log('saving %s_feature.csv' % file_name)
    unstack_final.to_csv('../data/%s_feature.csv' % file_name, index=None)
    #time.sleep(5)
    

























    