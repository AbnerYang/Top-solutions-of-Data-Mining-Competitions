
# coding: utf-8

# In[ ]:


import pandas as pd
from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
import time
import category_encoders as ce
import os
import copy

########### parameter ##############
mall_groups = {'m_690': ['m_690'],
               'm_5076': ['m_1175', 'm_1293', 'm_7168', 'm_3005', 'm_6337', 'm_5076', 'm_4094'],
               'm_909': ['m_1377', 'm_3839', 'm_2467', 'm_909', 'm_4422', 'm_8344', 'm_2224', 'm_4079'],
               'm_3871': ['m_1409', 'm_3871', 'm_7800', 'm_5892', 'm_5085'],
               'm_1920': ['m_4828', 'm_7523', 'm_1920', 'm_3739'],
               'm_9068': ['m_5825', 'm_822', 'm_1790', 'm_9068', 'm_1950', 'm_3832', 'm_2907', 'm_625'],
               'm_3517': ['m_7374', 'm_2415', 'm_2415', 'm_1621', 'm_2267', 'm_968', 'm_2182', 'm_1375', 'm_3517', 'm_2578', 'm_3054', 'm_6587', 'm_979', 'm_2878', 'm_615', 'm_1831', 'm_622'],
               'm_626': ['m_3112', 'm_4543', 'm_2009', 'm_4572', 'm_2270', 'm_5352', 'm_626', 'm_4341', 'm_1263', 'm_8093', 'm_4033', 'm_9054', 'm_7973', 'm_4495', 'm_4187', 'm_2333'],
               'm_1089': ['m_5154', 'm_4406', 'm_5767', 'm_3019', 'm_3528', 'm_3916', 'm_4515', 'm_3425', 'm_1089', 'm_623', 'm_4459', 'm_4011', 'm_3445', 'm_7994', 'm_4759', 'm_4168', 'm_7601', 'm_4923'],
               'm_2123': ['m_4548', 'm_6167', 'm_1021', 'm_8379', 'm_3501', 'm_2715', 'm_5529', 'm_2123', 'm_5810', 'm_4121', 'm_3313', 'm_1085', 'm_2058', 'm_6803']
               }


mall_params = {   
		'm_690': {'colsample_bytree': 0.5, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.8, 'eta': 0.09, 'objective': 'multi:softprob', 'max_depth': 6, 'gamma': 0.0, 'booster':'gbtree', 'silent':1,},
		'm_5076': {'colsample_bytree': 0.8, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.07, 'objective': 'multi:softprob', 'max_depth': 10, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_909': {'colsample_bytree': 0.7, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.09, 'objective': 'multi:softprob', 'max_depth': 10, 'gamma': 0.0, 'booster':'gbtree', 'silent':1,},
		'm_3871': {'colsample_bytree': 0.5, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.07, 'objective': 'multi:softprob', 'max_depth': 9, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_1920': {'colsample_bytree': 0.5, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.07, 'objective': 'multi:softprob', 'max_depth': 9, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_9068': {'colsample_bytree': 0.6, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.8, 'eta': 0.08, 'objective': 'multi:softprob', 'max_depth': 10, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_3517': {'colsample_bytree': 0.8, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.09, 'objective': 'multi:softprob', 'max_depth': 9, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_626': {'colsample_bytree': 0.5, 'eval_metric': 'merror', 'min_child_weight': 1, 'subsample': 0.8, 'eta': 0.08, 'objective': 'multi:softprob', 'max_depth': 7, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,},
		'm_1089': {'colsample_bytree': 0.5, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.6, 'eta': 0.08, 'objective': 'multi:softprob', 'max_depth': 9, 'gamma': 0.0, 'booster':'gbtree', 'silent':1,},
		'm_2123': {'colsample_bytree': 0.6, 'eval_metric': 'merror', 'min_child_weight': 0, 'subsample': 0.8, 'eta': 0.08, 'objective': 'multi:softprob', 'max_depth': 4, 'gamma': 0.01, 'booster':'gbtree', 'silent':1,}
		}
params_dlist = mall_params.values()
jitter_lists = {'colsample_bytree': np.arange(-0.2, 0.3, 0.05), 
                'subsample': np.arange(-0.2, 0.3, 0.05), 
                'eta': np.arange(-0.02, 0.03, 0.005), 
                'max_depth': np.arange(-2, 3, dtype=int), 
                'gamma': np.arange(0, 0.1, 0.02)}
def params_jitter(params):
    cparams = copy.deepcopy(params)
    for key, vals in jitter_lists.items():
        vallist = copy.deepcopy(vals)
        np.random.shuffle(vallist)
        if key != 'max_depth':
            if cparams[key] + vallist[0] > 1 or cparams[key] + vallist[0] <= 0:
                continue
        cparams[key] += vallist[0]
    return cparams
	
params_dict = {}
for mallg, params in mall_params.items():
    for mall in mall_groups[mallg]:
        params_dict[mall] = params
        
#shop = pd.read_csv("../data/mayi/ccf_first_round_shop_info.csv")
#train = pd.read_csv("../data/mayi/ccf_first_round_user_shop_behavior.csv")
#train = train.merge(shop[["shop_id", "mall_id"]], on="shop_id", how="left")
#test = pd.read_csv("../data/mayi/evaluation_public.csv")
#df = pd.concat([train, test])
#
#print "Processing date feature", time.strftime("%H:%M:%S",time.localtime())
#df['time_stamp'] = df.time_stamp.map(lambda t: time.strptime(t, u'%Y-%m-%d %H:%M'))
#df['moment'] = df.time_stamp.map(lambda t: t.tm_hour * 6 + t.tm_min / 10)
#df['day'] = df.time_stamp.map(lambda t: t.tm_wday + 1)
#df = pd.concat([df, pd.get_dummies(df['day']).add_prefix('wday_')], axis=1)
#df = df.drop(['day'], axis=1)
#print "Processing date feature done.", time.strftime("%H:%M:%S",time.localtime())

mall_list = params_dict.keys()
# mall_list = ["m_1293"]

#mall_count = 1
##result = pd.DataFrame()
#if not os.path.exists('../test/m_7168.csv'):
#    for mall in mall_list:
#        print "Preprocessing train mall.", mall, mall_count, "/ 97"
#        print time.strftime("%H:%M:%S",time.localtime())
#        mall_count += 1
#    
#        rows = []
#        wifi_col = {}
#        col = 0
#        wifi_count = defaultdict(int)
#        wifi_connect_count = defaultdict(int)
#        for index, row in df[df.mall_id == mall].reset_index(drop=True).iterrows():
#            for wifi in row.wifi_infos.split(";"):
#                ssid, dbm, connected = wifi.split("|")
#                if pd.isnull(row["shop_id"]) and not wifi_count.has_key(ssid):
#                    #过滤掉test中没有在train中出现过的wifi
#                    continue
#                dbm = int(dbm)
#                if not wifi_col.has_key(ssid):
#                    wifi_col[ssid] = col
#                    col += 1
#                row[ssid] = dbm
#                if connected == "true":
#                    row['connect'] = int(wifi_col[ssid])
#                    row['dbm'] = dbm
#                    row['is_connect'] = 1
#                    wifi_connect_count[ssid] += 1
#                wifi_count[ssid] += 1
#            rows.append(row)
#            
#    
#        drop_wifi_dict = {}
#        for ssid in wifi_count:
#            if wifi_count[ssid] == 1:
#                drop_wifi_dict[ssid] = 1
#        # 移除个人热点
#        # for ssid in wifi_connect_count:
#        #     if wifi_connect_count[ssid] == 1:
#        #         drop_wifi_dict[ssid] = 1
#        
#        tmp = pd.DataFrame(rows).drop(drop_wifi_dict.keys(), axis=1)
#    #     tmp['row_id'] = tmp['row_id'].astype('int')
#    #     tmp["connect"] = preprocessing.LabelEncoder().fit_transform(tmp["connect"].values)
#        
#        train_df = tmp[tmp.shop_id.notnull()].reset_index(drop=True)
#        test_df = tmp[tmp.shop_id.isnull()].reset_index(drop=True)
#        test_df["row_id"] = test_df["row_id"].astype("int")
#        train_df.to_csv('../train/%s.csv' % mall, index=None)
#        test_df.to_csv('../test/%s.csv' % mall, index=None)


rounds = {'m_1021': 108, 'm_1085': 178, 'm_1089': 202, 'm_1175': 224, 'm_1263': 101, 'm_1293': 118, 'm_1375': 218, 
        'm_1377': 205, 'm_1409': 208, 'm_1621': 155, 'm_1790': 197, 'm_1831': 177, 'm_1920': 149, 'm_1950': 154, 
        'm_2009': 128, 'm_2058': 97, 'm_2123': 110, 'm_2182': 127, 'm_2224': 181, 'm_2267': 170, 'm_2270': 116, 
        'm_2333': 84, 'm_2415': 159, 'm_2467': 270, 'm_2578': 159, 'm_2715': 79, 'm_2878': 206, 'm_2907': 177, 
        'm_3005': 255, 'm_3019': 172, 'm_3054': 144, 'm_3112': 125, 'm_3313': 96, 'm_3425': 127, 'm_3445': 141, 
        'm_3501': 95, 'm_3517': 201, 'm_3528': 137, 'm_3739': 258, 'm_3832': 201, 'm_3839': 231, 'm_3871': 197,
         'm_3916': 120, 'm_4011': 114, 'm_4033': 193, 'm_4079': 192, 'm_4094': 284, 'm_4121': 102, 'm_4168': 73, 
        'm_4187': 175, 'm_4341': 203, 'm_4406': 156, 'm_4422': 277, 'm_4459': 147, 'm_4495': 117, 'm_4515': 186,
         'm_4543': 193, 'm_4548': 173, 'm_4572': 173, 'm_4759': 149, 'm_4828': 121, 'm_4923': 126, 'm_5076': 175, 
        'm_5085': 132, 'm_5154': 166, 'm_5352': 136, 'm_5529': 78, 'm_5767': 124, 'm_5810': 136, 'm_5825': 177, 
        'm_5892': 170, 'm_615': 102, 'm_6167': 173, 'm_622': 132, 'm_623': 181, 'm_625': 228, 'm_626': 206, 
        'm_6337': 190, 'm_6587': 120, 'm_6803': 79, 'm_690': 227, 'm_7168': 248, 'm_7374': 156, 'm_7523': 232,
         'm_7601': 249, 'm_7800': 225, 'm_7973': 157, 'm_7994': 164, 'm_8093': 139, 'm_822': 211, 'm_8344': 168,
         'm_8379': 125, 'm_9054': 115, 'm_9068': 115, 'm_909': 162, 'm_968': 159, 'm_979': 170}

for k in range(500):
    print k, '/ 500'
    mall_count = 1
    use_params = []
    params = params_jitter(params_dlist[k % 10])
    for i, mall in enumerate(mall_list):
        for line in ['online', 'offline']:
            print "Preprocessing train mall.", mall, mall_count, "/ 97"
            print time.strftime("%H:%M:%S",time.localtime())
            mall_count += 1
            train_df = pd.read_csv('../train_%s/%s.csv' % (line, mall))
            test_df = pd.read_csv('../test_%s/%s.csv' % (line, mall))
            
            le = preprocessing.LabelEncoder()
            train_df["label"] = le.fit_transform(train_df.shop_id.values)
            test_df['label'] = np.nan
            enc = ce.LeaveOneOutEncoder(cols=["connect"])
            enc.fit(train_df, train_df['label'].values)
            train_df = enc.transform(train_df)
            test_df = enc.transform(test_df)
        #    params={
        #        'booster':'gbtree',
        #        'objective': 'multi:softprob',
        #        'max_depth': 7,
        #        'eta': 0.088,
        #        'seed':44,
        #        'silent':1,
        #        'colsample_bytree': 0.6,
        #        'subsample': 0.6,
        #        'min_child_weight': 1,
        #        'num_class': train_df.label.max()+1,
        #    }
            params['num_class'] = train_df.label.max()+1
            print params    
            
        #    print "xgb cv:", time.strftime("%H:%M:%S",time.localtime())
            feature = [x for x in train_df.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos', 'row_id']]
            
            dtrain = xgb.DMatrix(train_df[feature], label=train_df.label)
            #res = xgb.cv(params, dtrain, 1000, nfold=3, early_stopping_rounds=10, verbose_eval=10)
            #brs.append([mall, int(1.4 * len(res)), res["test-merror-mean"].iloc[-1]])
            #print brs[-1]
            print "xgb train:", time.strftime("%H:%M:%S",time.localtime())
            #model = xgb.train(params, dtrain, int(1.4 * len(res)), [(dtrain, 'train')], verbose_eval=5)
            model = xgb.train(params, dtrain, int(rounds[mall] * (1.05 + np.random.rand()/5)), [(dtrain, 'train')], verbose_eval=5)
        
            print "xgb test", time.strftime("%H:%M:%S",time.localtime())
            ptest = model.predict(xgb.DMatrix(test_df[feature]))
            r1, c1 = ptest.shape
            ptest = np.divide(ptest, np.sum(ptest, axis=1).reshape((r1, 1)))
            if i < 5:
                print np.sum(ptest, axis=1)[:10]
            ptest_label = ["" + str(le.inverse_transform(np.argmax(_))) for _ in ptest]
            ptest_prob = [np.max(_) for _ in ptest]
            test_df['shop_id'] = ptest_label
        #    result = pd.concat([result, test_df[["row_id", "shop_id"]]])
            row_id = test_df.row_id.values
            shops = [{'multi_prob': {"" + str(le.inverse_transform(sidx)): prob for sidx, prob in enumerate(_)}} for _ in ptest]
            print 'test done.'
            pd.DataFrame({"row_id": row_id, "shop_id": ptest_label, "prob": ptest_prob}).to_csv(("../res_%s/" % line) + str(mall) + ".csv", index=False)
            pd.DataFrame({"row_id": row_id, "candidate_dict": shops}).to_csv(("../res_%s/" % line) + str(mall) + "_prob.csv", index=False)
            print 'Saved results...'
    cparams = copy.deepcopy(params)
    use_params.append(cparams)
    #result.to_csv("./submission.csv", index=False)
    tmp = pd.DataFrame(use_params)
    tmp.to_csv("params_%d.csv" % k, index=None)
    #pd.DataFrame(brs).to_csv('./brs.csv', index=None)
    for line in ['online', 'offline']:
        print line, 'concat each mall prob file'
        files = os.listdir("../res_%s/" % line)
        nfiles = []
        lfiles = []
        for f in files:
            if 'prob' in f:
                nfiles.append(f)
            else:
                lfiles.append(f)
        
        probds, probs = [], []
        for f in lfiles:
            tmp = pd.read_csv(("../res_%s/" % line) + f)
            probs.append(tmp)
        for f in nfiles:
            tmp = pd.read_csv(("../res_%s/" % line) + f)
            probds.append(tmp)
        allprob = pd.concat(probds)
        predprob = pd.concat(probs)
        print line, 'saving concat prob file'
        allprob[['row_id', 'candidate_dict']].to_csv('../need_unstack/multi_prob_%s_%d.csv' % (line, k), index=None)
        predprob.to_csv('../need_unstack/pred_prob_%s_%d.csv' % (line, k), index=None)

