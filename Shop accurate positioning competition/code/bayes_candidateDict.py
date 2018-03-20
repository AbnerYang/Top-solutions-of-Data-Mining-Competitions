# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 10:06:44 2017

@author: 1
"""

import pandas as pd
import numpy as np
import time
import sys
import pickle
import os
import collections

#%% initialization
need_generate_neg = raw_input("need to generate negative set?(yes|no)")
skip_generate_tr = raw_input("need to skip generate train set?(yes|no)")
need_generate_all = raw_input("need all negative set?(yes|no)")
is_online = raw_input("running online?(yes|no)")
is_ok = raw_input("Test has passed, wanna eval in real offline test?(yes|no)")
is_pred = raw_input("How about predicting in test set?(yes|no)")

skip_generate_tr = (skip_generate_tr == 'yes')
need_generate_all = (need_generate_all == 'yes')
has_generate_neg = (need_generate_neg != 'yes')
#%% tools code by karon
def log(info):
    print time.strftime("[%Y-%m-%d %H:%M:%S]", time.gmtime()), info

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
    return np.max([1, int(wifi_force) + 120])
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
    user_shop[(user_shop.time_stamp >= "2017-08-17 00:00")].to_csv("../data/0817-0831.csv", index = None)
    evaluation = evaluation.rename(columns={'mall_id': 'shop_mall_id'})
    evaluation.to_csv("../data/new_evaluation_public.csv", index = None)



# preprocess
#if not os.path.exists('../data/0817-0831.csv') or not has_generate_neg:
#    log('preprocess')
#    preprocess()
#%% define train and test set
is_offline = (is_online != 'yes')


if not has_generate_neg:
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

if not has_generate_neg:
    log('get mall wifi id list')
    
        
    log('get joint wifi id list')
    wifi_ids = {}
    for set_key, data_set in data_sets.items():
        _, wifi_ids[set_key] = get_wifi_list(data_set)
    
    joint_wifi_id = wifi_ids['train'] + wifi_ids['test']
    tr = pd.Series(wifi_ids['train']).value_counts()
    te = pd.Series(wifi_ids['test']).value_counts()
    tr = tr.loc[tr > 3].index
    te = te.loc[te > 3].index
    tmp_joint = set(tr).intersection(te)
    tmp = pd.Series(joint_wifi_id).value_counts().reset_index()
    tmp.columns = ['wifi_id', 'counts']
    joint_wifi_id = set(tmp.loc[(tmp.counts > 5)].wifi_id).intersection(tmp_joint)
    #need = set(pd.read_csv('../data/wifi_id.csv', header=-1).iloc[:, 0].tolist())
    #joint_wifi_id = joint_wifi_id.intersection(need)
    print(len(joint_wifi_id))
# wifi_filter process
def clean_wifi_infos(df, wifi_ids):
    cdf = df.copy()
    def wifi_filter(wifis):
        clean = []
        showed_wifi = []
        tmp = wifis.split(';')
        for x in tmp:
            wid = x.split('|')[0]
            if wid in joint_wifi_id:# and wid not in showed_wifi:
                #showed_wifi.append(wid)
                clean.append(x)
        return str.join(';', clean)
    cdf['wifi_infos'] = cdf.wifi_infos.map(wifi_filter)
    cover_rec_rate = len(cdf.loc[cdf.wifi_infos != '', :]) / float(len(cdf))
    return cdf, cover_rec_rate


if not has_generate_neg:
    log('clean wifi infos')
    for set_key, data_set in data_sets.items():
        data_sets[set_key], _ = clean_wifi_infos(data_set, joint_wifi_id)
        print _
    tmp = data_sets['train'].groupby('shop_mall_id', 
                                       as_index=False).apply(lambda g: [g.shop_mall_id.iloc[0], 
                                                             str.join(';', g.wifi_infos.tolist())])
    mall_wifilist = {key: get_wifiids(val) for key, val in tmp.tolist()}
#%% construct mall_dict
# Dict Format
# wifi_info dict key structure {'mall_id': {'wifi_dict': wifi_dict, 'moment_dict': moment_dict}}
# wifi_info dict key structure [mall_id(97) -> 'wifi_dict' -> wifi_id(<<10W)] , 
# res form {
#           'prob_lookup_dict': {'force_level': {'is_link': prob-vector}}                     ## mall's shop prob vector [shop_id, prob] dict
#########   below may can be used to weight the prob_vector   ##########
#           'membership': mall's shop membership vector,                                        ## shop.linked_rate / next_shop.linked_rate
#           'force_sum': mall's shop force_list vector,
#           'mean_force': mall's shop mean_force vector,                                    ## mean_shoprec_force
#           'wifirec_force_sum': wifirec_force_sum
#           'wifirec_num': wifirec_num
#           'mean_wifirec_force': mean_wifirec_force
#           'mean_force_ratio': mall's shop mean_force_ratio vector,                        ## mean_shoprec_force / mean_wifirec_force
#           'cover_num': mall's shop cover_num vector,
#           'cover_rate': mall's shop cover_rate vector,
#           'linked_num': mall's shop linked_num vector,
#           'linked_rate': mall's shop linked_rate vector}
# wifi_info dict key structure [mall_id(97) -> 'moment_dict' -> moment_id(144)] , 
# res form [moment_prob shop_vector]
wifi_intervals = {'l1': [-40, 0], 'l2': [-50, -40], 'l3': [-55, -50], 'l4': [-60, -55],
                  'l5': [-65, -60], 'l6': [-70, -65], 'l7': [-75, -70], 'l8': [-80, -75],
                  'l9': [-86, -80], 'l10': [-92, -86], 'l11': [-120, -92]}
link_types = ['true', 'false']
is_skip_IO = True
def init_mall_shopvector(mall_id, val=1):
    shoplist = mall_shoplist[mall_id]
    shopvector = {shop_id: val for shop_id in shoplist}
    return shopvector

def init_mall_dict(assign_mall_id):
    def init_wifi_dict_content(mall_id):
        prob_lookup_dict = {force_level: 
                                {is_link: init_mall_shopvector(mall_id, 1e-5) for is_link in ['true', 'false']}
                            for force_level in ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10', 'l11']}
        wifi_dict = {'prob_lookup_dict': prob_lookup_dict,
                     'membership': init_mall_shopvector(mall_id, 0),
                     'force_sum': init_mall_shopvector(mall_id),
                     'mean_force': init_mall_shopvector(mall_id),
                     'wifirec_force_sum': 1,
                     'wifirec_num': 1,
                     'mean_wifirec_force': 1,
                     'mean_force_ratio': init_mall_shopvector(mall_id),
                     'cover_num': init_mall_shopvector(mall_id),
                     'cover_rate': init_mall_shopvector(mall_id),
                     'linked_num': init_mall_shopvector(mall_id),
                     'linked_rate': init_mall_shopvector(mall_id)}
        return wifi_dict
    def init_moment_dict_content(mall_id):
        moment_dict = {moment_id: init_mall_shopvector(mall_id) for moment_id in range(144)}
        return moment_dict
    
    mall_dict = {}
    mall_id = assign_mall_id
    wifilist = mall_wifilist[assign_mall_id]
    
    mall_dict[mall_id] = {}
    mall_dict[mall_id]['moment_dict'] = init_moment_dict_content(mall_id)
    mall_dict[mall_id]['wifi_dict'] = {}
    for wifi_id in wifilist:
        mall_dict[mall_id]['wifi_dict'][wifi_id] = init_wifi_dict_content(mall_id)
    return mall_dict

def get_force_level(wifi_force):
    level = 'l11'
    for key, interval in wifi_intervals.items():
        if wifi_force > interval[0] and wifi_force <= interval[1]:
            level = key
    return level

def mall_dict_content_extractor(tr, shop_id, mall_content):
    smtr = tr.loc[tr.shop_id == shop_id, :]
    lenG = len(smtr)
    L = range(lenG)
    for locat in L:
        rec = smtr.iloc[locat, :]
        wifis = map(lambda wifi: wifi.split('|') if wifi else ['miss', 0, 'unknown'], 
                    rec.wifi_infos.split(';'))
        moment_id = rec.moment
        moment_dict = mall_content['moment_dict']
        shopvector = moment_dict[moment_id]
        shopvector[shop_id] += 1
        for wifi_id, wifi_force, is_link in wifis:
            if wifi_id == 'miss':
                continue
            wifi_force = int(wifi_force)
            big_wifi_force = inverse_force(wifi_force)
            wifi_dict = mall_content['wifi_dict']
            wifi_dict[wifi_id]['wifirec_force_sum'] += big_wifi_force
            wifi_dict[wifi_id]['wifirec_num'] += 1
            shopvector = wifi_dict[wifi_id]['force_sum']
            shopvector[shop_id] += big_wifi_force
            shopvector = wifi_dict[wifi_id]['cover_num']
            shopvector[shop_id] += 1
            shopvector = wifi_dict[wifi_id]['linked_num']
            shopvector[shop_id] += 1
            
            prob_lookup_dict = wifi_dict[wifi_id]['prob_lookup_dict']
            force_level = get_force_level(wifi_force)
            shopvector = prob_lookup_dict[force_level][is_link]
            shopvector[shop_id] += 1

def reprocess(mall_id, mall_content):
    moment_dict = mall_content['moment_dict']
    ## mean_moment_shop_vector  (本来存在字典里就不用算了)
    moment_sum_shopvector = init_mall_shopvector(mall_id)
    for moment_id, shop_vector in moment_dict.items():
        for shop_id, val in shop_vector.items():
            moment_sum_shopvector[shop_id] += val
    for moment_id, shop_vector in moment_dict.items():
        for shop_id, val in moment_sum_shopvector.items():
            shop_vector[shop_id] = shop_vector[shop_id] / float(val)
        
    wifi_dict = mall_content['wifi_dict']    
    for wifi_id, wifi_dict_content in wifi_dict.items():
        prob_lookup_dict = wifi_dict_content['prob_lookup_dict']
        cover_num_shopvector = wifi_dict_content['cover_num']
        cover_rate_shopvector = wifi_dict_content['cover_rate']
        mean_force_shopvector = wifi_dict_content['mean_force']
        force_sum_shopvector = wifi_dict_content['force_sum']
        linked_num_shopvector = wifi_dict_content['linked_num']
        linked_rate_shopvector = wifi_dict_content['linked_rate']
        wifirec_force_sum = wifi_dict_content['wifirec_force_sum']
        wifirec_num = wifi_dict_content['wifirec_num']
        wifi_force = wifirec_force_sum / float(wifirec_num)
        wifi_dict_content['mean_wifirec_force'] = wifi_force
        mean_force_ratio_shopvector = wifi_dict_content['mean_force_ratio']
        
        membership_shopvector = wifi_dict_content['membership']
        
        for force_level in ['l%d' % (i+1) for i in range(11)]:
            for is_link in ['true', 'false']:
                prob_shopvector = prob_lookup_dict[force_level][is_link]
                for shop_id, val in moment_sum_shopvector.items():
                    prob_shopvector[shop_id] = prob_shopvector[shop_id] / float(val)
        total_cover_num = 0
        for shop_id, val in cover_num_shopvector.items():
            total_cover_num += val
            linked_rate_shopvector[shop_id] = linked_num_shopvector[shop_id] / float(val)
            
            shop_force = force_sum_shopvector[shop_id] / float(val)
            mean_force_shopvector[shop_id] = shop_force
            mean_force_ratio_shopvector[shop_id] = shop_force / float(wifi_force)
            
        for shop_id, val in cover_num_shopvector.items():
            cover_rate_shopvector[shop_id] = val / float(total_cover_num)
        
        cpy_linked_rate_shopvector = pd.DataFrame(dict(shop_id=linked_rate_shopvector.keys(), 
                                                       val=linked_rate_shopvector.values()))
        cpy_linked_rate_shopvector.sort_values(by='val', inplace=True, ascending=False)
        shop_ids = cpy_linked_rate_shopvector.shop_id.iloc[:-1].values
        v1 = cpy_linked_rate_shopvector.val.iloc[:-1].values
        v2 = cpy_linked_rate_shopvector.val[1:].values
        for idx, shop_id in enumerate(shop_ids):
            membership_shopvector[shop_id] = (1 + 100 * v1[idx]) / (1 + 100 * float(v2[idx]))
            #force_sum_shopvector[shop_id] = force_sum_shopvector[shop_id] / float(wifirec_force_sum) 
def construct_in_each_mall(tr, mall_id, mall_dict):
    mtr = tr.loc[tr.shop_mall_id == mall_id, :]
    mall_content_dict = mall_dict[mall_id]
    mall_shop_ids = init_mall_shopvector(mall_id)
    L = len(mall_shop_ids)
    for i, shop_id in enumerate(mall_shop_ids.keys()):
        if i % 50 == 0:
            finished_ratio = (i + 1) / float(L) * 100
            log("mall_id [%s] - shop_id [%s] - ShopCounter [%.1f%%]" % (mall_id, shop_id, finished_ratio))
        mall_dict_content_extractor(mtr, shop_id, mall_content_dict)
    log("reprocessing mall_id [%s]" % mall_id)
    reprocess(mall_id, mall_content_dict)
    
    
if not has_generate_neg:    
    if not os.path.exists('../data/mall_dict.pkl') or is_skip_IO:
        log("Construct mall_dict " + 'skipIO!' if is_skip_IO else '')
        mall_dict_list = []
        for i, mall_id in enumerate(mall_shoplist.keys()):
            #if mall_id != 'm_2907':
             #   print 'skip'
              #  continue
            log("----------------------------Mall No.%d----------------------------" % i)
            mall_dict = init_mall_dict(mall_id)
            construct_in_each_mall(data_sets['train'], mall_id, mall_dict)
            mall_dict_list.append((mall_id, mall_dict))
        print '%d Bytes' % sys.getsizeof(mall_dict_list)
        if not is_skip_IO:
            output = open("../data/mall_dict.pkl", "wb")
            log("saving mall dict")
            pickle.dump(mall_dict_list, output)
            output.close()
    else:
        inputf = open('../data/mall_dict.pkl', 'rb')
        mall_dict_list = pickle.load(inputf)
        inputf.close()
#%% Construct Truly Train and Test Set
is_testing = True
def cal_wifi_prob_shopvector(wifi_content_dict, moment_shopvector, force_level, is_link, mall_id):
    based_prob_shopvector = wifi_content_dict['prob_lookup_dict'][force_level]
    prob = init_mall_shopvector(mall_id, 0)
    for shop_id in prob.keys():
        prob[shop_id] = based_prob_shopvector['true'][shop_id] + based_prob_shopvector['false'][shop_id]
#    cover_num_shopvector = wifi_content_dict['cover_num']
#    cover_rate_shopvector = wifi_content_dict['cover_rate']
#    mean_force_shopvector = wifi_content_dict['mean_force']
#    linked_num_shopvector = wifi_content_dict['linked_num']
#    linked_rate_shopvector = wifi_content_dict['linked_rate']
#    mean_force_ratio_shopvector = wifi_content_dict['mean_force_ratio']
#    membership_shopvector = wifi_content_dict['membership']
#    
#    weighted_prob_shopvector = init_mall_shopvector(mall_id, 100)
#    for shop_id, based_prob in based_prob_shopvector.items():
#        weighted_prob_shopvector[shop_id] = weighted_prob_shopvector[shop_id] *     \
#                                            cover_rate_shopvector[shop_id] *        \
#                                            mean_force_ratio_shopvector[shop_id] *  \
#                                            membership_shopvector[shop_id] *        \
#                                            linked_rate_shopvector[shop_id]
    return based_prob_shopvector[is_link]#wifi_content_dict['force_sum']

def WiFiInfo2Candidates(wifiInfo, mall_id, moment_id):
    wifis = map(lambda wifi: wifi.split('|') if wifi else ['miss', 0, 'unknown'], 
                        wifiInfo.split(';'))
    for mall, mall_dict in mall_dict_list:
        if mall_id == mall:
            mall_content_dict = mall_dict[mall_id]
            break
    cumprod_prob_shopvector = init_mall_shopvector(mall_id, 1000)
    cumprod_meanprob_shopvector = init_mall_shopvector(mall_id, 1000)
    Counter = init_mall_shopvector(mall_id, 0)
    #orMax_prob_shopvector = init_mall_shopvector(mall_id, 0)
    
    wifi_dict = mall_content_dict['wifi_dict']
    moment_shopvector = mall_content_dict['moment_dict'][moment_id]
    is_miss_wifiinfo = False  
    wifiL = len(wifis)
    wifiIds = wifi_dict.keys()
    wifis = pd.DataFrame(wifis, columns=['wid', 'level', 'link']).sort_values('level', ascending=False).values.tolist()
    for c, (wifi_id, wifi_force, is_link) in enumerate(wifis):
        if wifi_id == 'miss' or (wifiL == 1 and wifi_id not in wifiIds):
            is_miss_wifiinfo = True
            break
        if wifi_id not in wifiIds:
            break
        force_level = get_force_level(int(wifi_force))
        wifi_content_dict = wifi_dict[wifi_id]
        wifi_mean_force = wifi_content_dict['mean_wifirec_force']
        wifi_ratio = inverse_force(wifi_force) / float(wifi_mean_force)
        prob_shopvector = cal_wifi_prob_shopvector(wifi_content_dict, moment_shopvector, force_level, is_link, mall_id)
        for shop_id, val in prob_shopvector.items():
            cumprod_prob_shopvector[shop_id] = cumprod_prob_shopvector[shop_id] * float(val)
            cumprod_meanprob_shopvector[shop_id] = cumprod_meanprob_shopvector[shop_id] * float(val)
            if val != 1:
                Counter[shop_id] = Counter[shop_id] + 1.0
    #        orMax_prob_shopvector[shop_id] = float(np.max([orMax_prob_shopvector[shop_id] * wifi_ratio, val]))
    for shop_id, val in cumprod_meanprob_shopvector.items():
        if Counter[shop_id] == 0:
            continue
        cumprod_meanprob_shopvector[shop_id] = pow(val, 1.0 / Counter[shop_id])
    for shop_id, val in moment_shopvector.items():
        if is_miss_wifiinfo:
            cumprod_prob_shopvector[shop_id] = val
            cumprod_meanprob_shopvector[shop_id] = val
   #         orMax_prob_shopvector[shop_id] = val
        else:
            cumprod_prob_shopvector[shop_id] = np.sqrt(cumprod_prob_shopvector[shop_id] * float(val))
            cumprod_meanprob_shopvector[shop_id] = np.sqrt(cumprod_meanprob_shopvector[shop_id] * float(val))
  #          orMax_prob_shopvector[shop_id] = np.sqrt(orMax_prob_shopvector[shop_id] * float(val))
    
    return {'cumprod': cumprod_prob_shopvector,'cumprod_mean': cumprod_meanprob_shopvector} #'cumsum': orMax_prob_shopvector}
global_Counter = 0    
def eval_neg(df_cand, topK, set_key='train'):
    cpy_df_cand = df_cand.copy()
    global global_Counter
    global_Counter = 0
    lenG = float(len(df_cand))
    def get_topK_shoplists(candidates):
        global global_Counter
        global_Counter = global_Counter + 1
        if global_Counter % 5000 == 0:
            finished_ratio = (global_Counter + 1) / lenG * 100
            log("Gnerating topK [%s] - RecCounter [%.1f%%]" % (set_key, finished_ratio))
        if type(candidates) is str:
            candidates = eval(candidates)
        list_dict = {key: {'shop_id': {rank: 'miss' for rank in range(topK)},
                                 'val': {rank: 0 for rank in range(topK)}} 
                     for key in candidates.keys()}
                     
        nrows_2cols = pd.DataFrame(candidates)
        for key in list_dict.keys():
            tmp_series = nrows_2cols[key].sort_values(ascending=False).iloc[:topK].copy()
            index = tmp_series.index
            values = tmp_series.values
            LenG = len(tmp_series)
            for idx in range(LenG):
                list_dict[key]['shop_id'][idx] = index[idx]
                list_dict[key]['val'][idx] = values[idx]
        return list_dict
    
    log("generating top%d Candidates on %s set" % (topK, set_key))
    cpy_df_cand['topK_Candidates'] = cpy_df_cand.candidate_dict.map(get_topK_shoplists)
    log("saving new_df with top%d Candidates on %s set" % (topK, set_key))
    #cpy_df_cand.to_csv("../data/%s_top%d.csv" % (set_key, topK), index=None)
    
    prob_type_keys = cpy_df_cand.topK_Candidates.iloc[0].keys()
    if set_key == 'train':
        log("evaluating on train set")
        LenG = len(cpy_df_cand)
        res_TupleDict = []
        for idx in range(LenG):
            if idx % 5000 == 0:
                finished_ratio = (idx + 1) / float(LenG) * 100
                log("Evaluating - RecCounter [%.1f%%]" % (finished_ratio))
            topK_Candidates = cpy_df_cand.topK_Candidates.iloc[idx]
            truth_shop = cpy_df_cand.shop_id.iloc[idx]
            tuple_dict = {key: ([1 if truth_shop == topK_Candidates[key]['shop_id'][rank] else 0 
                                     for rank in range(topK)], 
                                 1 if truth_shop == topK_Candidates[key]['shop_id'][0] else 0) 
                          for key in topK_Candidates.keys()}
            res_TupleDict.append(tuple_dict)
        cpy_df_cand['eval_res_TupleDict'] = pd.Series(res_TupleDict, index=cpy_df_cand.index)
        # evaluate
        
        res_dict = {key: {'iscovers': [], 'iscorrect': []} for key in prob_type_keys}
        for tuple_dicts in cpy_df_cand.eval_res_TupleDict.values:
            for key, (iscovers, iscorrect) in tuple_dicts.items():
                res_dict[key]['iscovers'].append(iscovers)
                res_dict[key]['iscorrect'].append(iscorrect)
        for i in range(topK):
            topk = i + 1
            for key, cont in res_dict.items():
                for rtype, rlist in cont.items():
                    if rtype == 'iscovers':
                        rlist = [np.sum(ele[:topk]) for ele in rlist]
                    else:
                        continue
                    log("top%d - %s - %s - [%.4f] - [/%d]" % 
                            (topk, key, rtype, np.mean(rlist), len(rlist)))
        
        #log("saving new_df generated by evaluating process")
        #cpy_df_cand.to_csv("../data/eval_inter_result_top%d.csv" % topK, index=None)
        
    elif set_key == 'test':
        log("predicting on test set")
        LenG = len(cpy_df_cand)
        pred_shop_list = []
        for idx in range(LenG):
            if idx % 5000 == 0:
                finished_ratio = (idx + 1) / float(LenG) * 100
                log("Evaluating - RecCounter [%.1f%%]" % (finished_ratio))
            topK_Candidates = cpy_df_cand.topK_Candidates.iloc[idx]
            pred_shop_prod1, pred_shop_prod2 = topK_Candidates['cumprod']['shop_id'][0], topK_Candidates['cumprod']['shop_id'][1]
            pred_shop_sum1,pred_shop_sum2  = topK_Candidates['cumprod_mean']['shop_id'][0], topK_Candidates['cumprod_mean']['shop_id'][1]
            pred_shops = [pred_shop_prod1, pred_shop_prod2, pred_shop_sum1,pred_shop_sum2]
            hit_nums_vals = collections.Counter(pred_shops).values()
            hit_nums_dict = collections.Counter(pred_shops)
			
            hit_max, hit_min = np.max(hit_nums_vals), np.min(hit_nums_vals)
            
            if hit_max == 2 and hit_min == 2:
                pred_shop = pred_shop_prod1
            elif hit_max == 2 and hit_min == 1:
                for key, val in hit_nums_dict.items():
                    if val == 2:
                        pred_shop = key
            else:
                pred_shop = pred_shop_prod1
            
            pred_shop_list.append(pred_shop)
        cpy_df_cand['shop_id'] = pd.Series(pred_shop_list, index=cpy_df_cand.index)
        
        pred_df = cpy_df_cand.loc[:, ['row_id', 'shop_id']].copy()
        log("saving pred_res generated by evaluating process")
        pred_df.to_csv("../data/first_step_pred_res.csv", index=None)
		
        for No_k in range(1):
            cpy_df_cand['shop_id'] = cpy_df_cand.topK_Candidates.map(lambda shop_dict: shop_dict['cumprod']['shop_id'][No_k])
            pred_df = cpy_df_cand.loc[:, ['row_id', 'shop_id']].copy()
            pred_df.to_csv("../data/pred_res_No%d.csv" % (No_k + 1), index=None)
            
            cpy_df_cand['shop_id'] = cpy_df_cand.topK_Candidates.map(lambda shop_dict: shop_dict['cumprod_mean']['shop_id'][No_k])
            pred_df = cpy_df_cand.loc[:, ['row_id', 'shop_id']].copy()
            pred_df.to_csv("../data/pred_res_No%d_mean.csv" % (No_k + 1), index=None)



#train set shrinkage

#data_sets['train'] = data_sets['train'].iloc[:10000, :]


log('Testing' if is_testing else 'Predicting')
set_list = ['train','test']
if skip_generate_tr:
    set_list = ['test']
if not has_generate_neg:
    log('using mall_dict constructing candidates shop vector [%s]' % ('ALL' if need_generate_all else 'PART'))
    if not need_generate_all:
        for set_key in set_list:
            #data_sets[set_key] = data_sets[set_key].loc[data_sets[set_key].shop_mall_id == 'm_7168', :]                        
            data_sets[set_key] = data_sets[set_key].iloc[:10000, :]
    for set_key in set_list:
        data_set = data_sets[set_key]
        if 'row_id' not in data_set.columns:
            data_set['row_id'] = np.arange(len(data_set))
        LenG = len(data_set)
        series_dict = {}
        for idx in range(LenG):
            if idx % 5000 == 0:
                finished_ratio = (idx + 1) / float(LenG) * 100
                log("data_set [%s] - ShopCounter [%.1f%%]" % (set_key, finished_ratio))
            wifi_infos = data_set.wifi_infos.iloc[idx]
            mall_id = data_set.shop_mall_id.iloc[idx]
            moment_id = data_set.moment.iloc[idx]
            index = data_set.index[idx]
            series_dict[index] = WiFiInfo2Candidates(wifi_infos, mall_id, moment_id)
        
        log('saving data set with candidates shopdict')
        data_set['candidate_dict'] = pd.Series(series_dict)
        need_cols = (['row_id'] if 'row_id' in data_set.columns else []) + ['candidate_dict']
        data_set.loc[:, need_cols].to_csv("../bayes_%s_candidate.csv" % ('offline' if is_offline else 'online'), index = None)
        time.sleep(3)
    #log("negative set has generated, delete mall_dict")
    #del mall_dict_list
    time.sleep(3)
else:
    log("reloading data set with candidates shopdict")
    data_sets = {}
    # candidates 会变成string 而不是dict
    for set_key in set_list:
        data_sets[set_key] = pd.read_csv("../data/final_%s.csv" % set_key)
        
log('evaluating candidates shop vector')
eval_set = data_sets['train']
if is_testing:
    if not skip_generate_tr:
        log('train set testing')
        ptr_eval_set = eval_set.iloc[:6000, :].copy()
        eval_neg(ptr_eval_set, 10)
    if is_offline:
        log('test set testing')
        ptr_eval_set = data_sets['test'].iloc[:6000, :].copy()
        eval_neg(ptr_eval_set, 10)
        
if is_ok == "yes":
#    log("evaluating train set")
#    eval_neg(eval_set, 10)
    if 'shop_id' in data_sets['test'].columns:
    	log("----------real scene test set evaluate result----------")
    	eval_neg(data_sets['test'], 10)
if is_pred == "yes":
    eval_neg(data_sets['test'], 10, 'test')



