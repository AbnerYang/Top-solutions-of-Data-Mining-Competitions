# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:27:20 2017

@author: 1
"""
#%%
import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances
import time
import os

is_online = 'no'
files = [('online_train', 'tr_new_prob', 'skip', '0817-0831'), ('online_test', 'skip', '0801-0831', '0901-0914')]

has_generate_prefeature_file = False
#set_list = ['local']#, 'online']
cate_dict = {'c_1': 11, 'c_10': 43, 'c_11': 42, 'c_112': 20, 'c_12': 47, 'c_127': 58, 'c_128': 21, 'c_129': 22, 'c_13': 46,
             'c_130': 51, 'c_131': 50, 'c_132': 49, 'c_133': 48, 'c_134': 54, 'c_135': 53, 'c_136': 52, 'c_14': 39, 'c_15': 38,
             'c_16': 41, 'c_17': 40, 'c_18': 36, 'c_19': 19, 'c_2': 13, 'c_20': 4, 'c_21': 5, 'c_22': 6, 'c_23': 7, 'c_24': 0, 
             'c_25': 1, 'c_26': 2, 'c_27': 3, 'c_28': 9, 'c_29': 10, 'c_3': 12, 'c_30': 37, 'c_33': 35, 'c_34': 34, 'c_345': 23, 
             'c_36': 33, 'c_38': 45, 'c_39': 44, 'c_4': 15, 'c_40': 61, 'c_41': 62, 'c_42': 59, 'c_43': 60, 'c_44': 57, 'c_45': 24, 
             'c_46': 55, 'c_47': 56, 'c_48': 65, 'c_49': 66, 'c_5': 14, 'c_50': 26, 'c_51': 25, 'c_52': 28, 'c_53': 27, 'c_54': 30, 
             'c_55': 29, 'c_56': 31, 'c_59': 32, 'c_6': 17, 'c_62': 8, 'c_7': 16, 'c_8': 64, 'c_82': 63, 'c_9': 18}

malls = ['m_4079', 'm_4923', 'm_6337', 'm_9068', 'm_5085', 'm_6587', 'm_7374', 'm_1409', 'm_822', 
        'm_3739', 'm_2224', 'm_4187', 'm_622', 'm_2058', 'm_615', 'm_7973', 'm_5352', 'm_5767', 
        'm_690', 'm_5892', 'm_2333', 'm_4033', 'm_4341', 'm_4828', 'm_979', 'm_1375', 'm_5154', 
        'm_5825', 'm_3839', 'm_3112', 'm_623', 'm_2267', 'm_1085', 'm_3517', 'm_9054', 'm_2009', 
        'm_2270', 'm_6167', 'm_4548', 'm_1175', 'm_8093', 'm_909', 'm_2123', 'm_3832', 'm_3871', 
        'm_1950', 'm_1293', 'm_1263', 'm_4094', 'm_1920', 'm_7523', 'm_626', 'm_1621', 'm_4121', 
        'm_7168', 'm_3445', 'm_3501', 'm_2467', 'm_4168', 'm_7994', 'm_3425', 'm_5076', 'm_7800', 
        'm_625', 'm_3916', 'm_2907', 'm_3054', 'm_3005', 'm_1377', 'm_4011', 'm_1831', 'm_4459', 
        'm_2182', 'm_4572', 'm_2415', 'm_1021', 'm_1790', 'm_3528', 'm_8344', 'm_7601', 'm_968', 
        'm_8379', 'm_4495', 'm_4406', 'm_2578', 'm_4422', 'm_2715', 'm_4759', 'm_3019', 'm_4543', 
        'm_2878', 'm_1089', 'm_5529', 'm_3313', 'm_6803', 'm_5810', 'm_4515']
#%%
ring_edges = [7, 5, 3]
#knn_features = ['E_row_shop_RingCount', 'E_row_shop_RingRate', 'M_row_shop_RingCount', 'M_row_shop_RingRate']
# mall by mall
def r2df(r, prefix):
    tmp3 = r[r <= ring_edges[0]]
    tmp2 = tmp3[tmp3 <= ring_edges[1]]
    tmp1 = tmp2[tmp2 <= ring_edges[2]]
    res_df = pd.concat([pd.Series(tmp3.index).value_counts(), 
                        pd.Series(tmp2.index).value_counts(), 
                        pd.Series(tmp1.index).value_counts()], axis=1).fillna(1e-5)
    res_df.columns = [prefix + 'inRing_ShopCount_%d' % e for e in ring_edges]
    
    return res_df.T.stack()
shop_info = pd.read_csv('../data/shop_info.csv')
shop_info = shop_info.add_prefix('Fshop_')
shop_info.rename(columns = {'Fshop_shop_id': 'candidate_shop_id'}, inplace=True)
base = '../data/'
for filenames in files:
    if filenames[2] == 'skip':
        print('skip %s' % filenames[0])
        continue
    print('loading %s' % filenames[0])
    data = pd.read_csv('../data/%s.csv' % filenames[0])
    data = data.merge(shop_info[['candidate_shop_id', 'Fshop_mall_id']], on='candidate_shop_id', how='left').rename(columns={'Fshop_mall_id': 'shop_mall_id'})
#    data.row_id = data.row_id.astype(str)
    
    train = pd.read_csv(base + filenames[2] + '.csv')
    test = pd.read_csv(base + filenames[3] + '.csv')
    scalar = 1138015.0 / len(train)
    train.rename(columns={'shop_id': 'candidate_shop_id'}, inplace=True)
    train = train.merge(shop_info, on='candidate_shop_id', how='left')
    if 'shop_id' in test.columns:
        test.rename(columns={'shop_id': 'candidate_shop_id'}, inplace=True)
        test = test.merge(shop_info, on='candidate_shop_id', how='left')
    else:
        test.rename(columns={'mall_id' : 'Fshop_mall_id'}, inplace=True)
    L = len(test)
#    test['row_id'] = map(str, np.arange(L))
    
    mdatas = []
    for c, mall_id in enumerate(malls):
        print('No.%d mall [%s]' % (c + 1, mall_id))
        mdata = data.loc[data.shop_mall_id == mall_id, :]
        
        mtrain = train.loc[train.Fshop_mall_id == mall_id, ['candidate_shop_id', 'longitude', 'latitude']]
        mtest = test.loc[test.Fshop_mall_id == mall_id, ['row_id', 'longitude', 'latitude']]
        
        euclidean_mat = pairwise_distances(111000 * mtest.iloc[:, 1:].values, 111000 * mtrain.iloc[:, 1:].values, metric='euclidean')
        manhattan_mat = pairwise_distances(111000 * mtest.iloc[:, 1:].values, 111000 * mtrain.iloc[:, 1:].values, metric='manhattan')
        ## euclidean feature
        print('------- Euclidean')
        distdf = pd.DataFrame(euclidean_mat, index=mtest.row_id, columns=mtrain.candidate_shop_id)
        # get count 
        row_shop_RingCount = distdf.apply(lambda r: r2df(r, 'E_'), axis=1).fillna(1e-6)
        # get rate
        row_shop_RingRate = []
        col_names = []
        for col in ['E_inRing_ShopCount_%d' % e for e in ring_edges]:
            row_shop_RingRate.append(row_shop_RingCount.loc[:, col].divide(row_shop_RingCount.loc[:, col].sum(1), 
                                     axis=0).stack().map(lambda v: np.exp(v + 5) - 140))
            col_names.append(col.replace('Count', 'Rate'))
        row_shop_RingRate = pd.concat(row_shop_RingRate, axis=1)
        row_shop_RingRate.columns = col_names
        row_shop_RingCount = row_shop_RingCount.stack() * scalar
        
        row_shop_RingRate = row_shop_RingRate.reset_index()
        row_shop_RingRate.rename(columns={'level_1': 'candidate_shop_id'}, inplace=True)
        row_shop_RingCount = row_shop_RingCount.reset_index()
        row_shop_RingCount.rename(columns={'level_1': 'candidate_shop_id'}, inplace=True)
        
        c_cols = row_shop_RingCount.columns
        mdata = mdata.merge(row_shop_RingCount, on=['row_id', 'candidate_shop_id'], how='left')
#        mdata.loc[:, c_cols] = mdata.loc[:, c_cols].apply(lambda c: c.fillna(c.min()), axis=0)
        r_cols = row_shop_RingRate.columns
        mdata = mdata.merge(row_shop_RingRate, on=['row_id', 'candidate_shop_id'], how='left')
#        mdata.loc[:, r_cols] = mdata.loc[:, r_cols].apply(lambda c: c.fillna(c.min()), axis=0)
        
        ## manhattan feature
#        print('------- Manhattan')
#        distdf = pd.DataFrame(manhattan_mat, index=mtest.row_id, columns=mtrain.candidate_shop_id)
#        # get count 
#        row_shop_RingCount = distdf.apply(lambda r: r2df(r, 'M_'), axis=1).fillna(1e-6)
#        # get rate
#        row_shop_RingRate = []
#        col_names = []
#        for col in ['M_inRing_ShopCount_%d' % e for e in ring_edges]:
#            row_shop_RingRate.append(row_shop_RingCount.loc[:, col].divide(row_shop_RingCount.loc[:, col].sum(1), 
#                                     axis=0).stack().map(lambda v: np.exp(v + 5) - 140))
#            col_names.append(col.replace('Count', 'Rate'))
#        row_shop_RingRate = pd.concat(row_shop_RingRate, axis=1)
#        row_shop_RingRate.columns = col_names
#        row_shop_RingCount = row_shop_RingCount.stack() * scalar
#        
#        row_shop_RingRate = row_shop_RingRate.reset_index()
#        row_shop_RingRate.rename(columns={'level_1': 'candidate_shop_id'}, inplace=True)
#        row_shop_RingCount = row_shop_RingCount.reset_index()
#        row_shop_RingCount.rename(columns={'level_1': 'candidate_shop_id'}, inplace=True)
#        
#        c_cols = row_shop_RingCount.columns
#        mdata = mdata.merge(row_shop_RingCount, on=['row_id', 'candidate_shop_id'], how='left')
#        mdata.loc[:, c_cols] = mdata.loc[:, c_cols].apply(lambda c: c.fillna(c.min()), axis=0)
#        r_cols = row_shop_RingRate.columns
#        mdata = mdata.merge(row_shop_RingRate, on=['row_id', 'candidate_shop_id'], how='left')
#        mdata.loc[:, r_cols] = mdata.loc[:, r_cols].apply(lambda c: c.fillna(c.min()), axis=0)
        mdatas.append(mdata)
        
    final_data = pd.concat(mdatas, axis=0)
    print('saving knn-preprocess result')
    final_data.to_csv('../data/knn_%s.csv' % filenames[0], index=None)
