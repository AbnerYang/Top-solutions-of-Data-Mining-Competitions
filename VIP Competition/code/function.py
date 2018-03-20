#-*- encoding:utf-8 -*-
import numpy as np 
import pandas as pd
import time 
import copy
import scipy as sp
import os
import warnings
import random
warnings.filterwarnings("ignore")

def Log(info):
	print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+' '+str(info)

def getWeek(df):
    df = df.sort_values('date',ascending=False)
    dateDic = {}
    weeks = 0
    daycount = -1
    data = df.date.values
    week = []
    for d in data:
        if dateDic.has_key(d) == False:
            if daycount == 6:
                daycount = 0
                weeks += 1
                print weeks
            else:
                daycount += 1
            dateDic[d] = weeks
        week.append(dateDic[d])
    df['week'] = week
    return df

def preprocess():
	Log('read train data..')
	df = pd.read_table('../data/user_action_train.txt',sep='\t',header=None)
	df.columns = ['uid','spu_id','label','date']
	Log('get week info..')
	df = getWeek(df)
	print np.unique(df.date)
	df.to_csv('../data/local_train.csv',index=0)

	Log('read test data..')
	df2 = pd.read_table('../data/user_action_test_items.txt',sep='\t',header=None)
	df2.columns = ['uid','spu_id','label']
	df2.to_csv('../data/online_test.csv',index=0)

	Log('read goods data..')
	df3 = pd.read_table('../data/goods_train.txt',sep='\t',header=None)
	df3.columns=['spu_id','brand_id','cat_id']
	df3.to_csv('../data/goods.csv',index=0)

def rename(df,old,new):
    df = df.rename(columns = {old:new})
    return df

def getUserFeature(df,users,name='spu_id',num_weeks=1):
    weeks = '_'+str(num_weeks)
    ##用户点击数目
    user = users[['uid']]
    mydf = df.groupby(['uid']).size().reset_index()
    mydf = rename(mydf,0,'user_click_'+name+'_counts'+weeks)
    user = pd.merge(user,mydf,on='uid',how='left')
    
    ##用户购买数目/比率
    mydf = df.groupby(['uid',])['label'].sum().reset_index()
    mydf = rename(mydf,'label','user_buy_'+name+'_counts'+weeks)
    user = pd.merge(user,mydf,on=['uid'],how='left')
    user['user_buy_'+name+'_rate'+weeks] = user['user_buy_'+name+'_counts'+weeks] / user['user_click_'+name+'_counts'+weeks]
    user['user_not_buy_'+name+'_rate'+weeks] = 1 - user['user_buy_'+name+'_rate'+weeks] 
    user['user_not_buy_'+name+'_counts'+weeks] =  user['user_click_'+name+'_counts'+weeks] - user['user_buy_'+name+'_counts'+weeks]
    ###用户点击商品种类数/比率
    mydf = df.groupby(['uid',name]).size().reset_index()
    mydf = mydf.groupby(['uid']).size().reset_index()
    mydf = rename(mydf,0,'user_click_'+name+'_kinds'+weeks)
    
    pos = df[df.label==1]
    mydf1 = pos.groupby(['uid',name]).size().reset_index()
    mydf1 = mydf1.groupby(['uid']).size().reset_index()
    mydf1 = rename(mydf1,0,'user_buy_'+name+'_kinds'+weeks)
    mydf = pd.merge(mydf,mydf1,on=['uid'],how='left')
    mydf['user_buy_kind_'+name+'_rate'+weeks] = mydf['user_buy_'+name+'_kinds'+weeks] / mydf['user_click_'+name+'_kinds'+weeks]
    mydf['user_not_buy_kind_'+name+'_rate'+weeks] = 1 - mydf['user_buy_kind_'+name+'_rate'+weeks]
    mydf['user_not_buy_kind_'+name+'counts'+weeks] = mydf['user_click_'+name+'_kinds'+weeks] -  mydf['user_buy_'+name+'_kinds'+weeks]
    user = pd.merge(user,mydf,on=['uid'],how='left').fillna(0)
    return user

def getUserFeature2(df,user,week,num_weeks):
    weeks = str(num_weeks)
    pos = df[df.label==1] 
    ##用户平均每品牌购买数
    mydf = pos.groupby(['uid','brand_id']).size().reset_index()
    mydf = mydf.groupby('uid')[0].mean().reset_index()
    mydf = rename(mydf,0,'user_avg_buy_each_brand'+weeks)
    user = pd.merge(user,mydf,on='uid',how='left')
    ##用户平均每类别购买数
    mydf = pos.groupby(['uid','cat_id']).size().reset_index()
    mydf = mydf.groupby('uid')[0].mean().reset_index()
    mydf = rename(mydf,0,'user_avg_buy_each_cat'+weeks)
    user = pd.merge(user,mydf,on='uid',how='left')
    
    ##用户购买/点击历史次数
    namelst = ['spu_id','brand_id','cat_id']
    for name in namelst:     
        mydf = df.groupby(['uid',name]).size().reset_index()
        mydf = rename(mydf,0,'user_history_click_'+name+'_counts'+weeks)
        mydf1 = pos.groupby(['uid',name]).size().reset_index()
        mydf1 = rename(mydf1,0,'user_history_buy_'+name+'_counts'+weeks)
        mydf = pd.merge(mydf,mydf1,on=['uid',name],how='left').fillna(0)
        mydf['user_history_buy_'+name+'_rate'+weeks] = mydf['user_history_buy_'+name+'_counts'+weeks]/mydf['user_history_click_'+name+'_counts'+weeks]
        mydf['user_history_not_buy_'+name+'_rate'+weeks] =1 - mydf['user_history_buy_'+name+'_rate'+weeks] 
        mydf['user_history_not_buy_'+name+'_counts'+weeks] = mydf['user_history_click_'+name+'_counts'+weeks] - mydf['user_history_buy_'+name+'_counts'+weeks]
        user = pd.merge(user,mydf,on=['uid',name],how='left').fillna(0)
        
    mydf1 = pos.groupby(['uid','week']).size().reset_index()
    mydf_min = mydf1.groupby('uid')['week'].min().reset_index()
    mydf_min['week'] = mydf_min['week'] - week
    mydf_min = rename(mydf_min,'week','last_buy_week'+weeks)
    user = pd.merge(user,mydf_min,on='uid',how='left').fillna(-1)

    return user

def getGoodFeature(df,spu,name,num_weeks):
    weeks = str(num_weeks)
    ##商品被点击数目
    mydf = df.groupby([name]).size().reset_index()
    mydf = rename(mydf,0,name+'_click_counts'+weeks)
    spu = pd.merge(spu,mydf,on=name,how='left')
    
    ##商品被购买数目/比率
    mydf = df.groupby([name])['label'].sum().reset_index()
    mydf = rename(mydf,'label',name+'_buy_counts'+weeks)
    spu = pd.merge(spu,mydf,on=[name],how='left')
    spu[name+'_buy_rate'+weeks] = spu[name+'_buy_counts'+weeks] / spu[name+'_click_counts'+weeks]
    ###商品被点击/购买用户数/比率
    mydf = df.groupby(['uid',name]).size().reset_index()
    mydf = mydf.groupby([name]).size().reset_index()
    mydf = rename(mydf,0,name+'_click_users_counts'+weeks)
    
    pos = df[df.label==1]
    mydf1 = pos.groupby(['uid',name]).size().reset_index()
    mydf1 = mydf1.groupby([name]).size().reset_index()
    mydf1 = rename(mydf1,0,name+'_buy_users_counts'+weeks)
    mydf = pd.merge(mydf,mydf1,on=[name],how='left')
    mydf[name+'_buy_user_rate'+weeks] = mydf[name+'_buy_users_counts'+weeks] / mydf[name+'_click_users_counts'+weeks]
    spu = pd.merge(spu,mydf,on=[name],how='left').fillna(0)
    return spu

def getTrickFeature(df):
    mydf = df.groupby('uid').size().reset_index()
    mydf = rename(mydf,0,'user_this_week_click')
    df = pd.merge(df,mydf,on='uid',how='left')
    
    mydf = df.groupby(['spu_id']).size().reset_index()
    mydf = rename(mydf,0,'spu_this_week_click')
    df = pd.merge(df,mydf,on='spu_id',how='left')
    
    mydf = df.groupby(['brand_id']).size().reset_index()
    mydf = rename(mydf,0,'brand_this_week_click')
    df = pd.merge(df,mydf,on='brand_id',how='left')
    
    mydf = df.groupby(['cat_id']).size().reset_index()
    mydf = rename(mydf,0,'cat_this_week_click')
    df = pd.merge(df,mydf,on='cat_id',how='left')
    
    mydf = df.groupby(['cat_id','uid']).size().reset_index()
    mydf = rename(mydf,0,'user_cat_this_week_click')
    df = pd.merge(df,mydf,on=['cat_id','uid'],how='left')
    
    mydf = df.groupby(['brand_id','uid']).size().reset_index()
    mydf = rename(mydf,0,'user_brand_this_week_click')
    df = pd.merge(df,mydf,on=['uid','brand_id'],how='left')
    
    mydf = df.groupby(['brand_id','spu_id']).size().reset_index()
    mydf = mydf.groupby(['brand_id']).size().reset_index()
    mydf = rename(mydf,0,'brand_spu_kinds_this_week')
    df = pd.merge(df,mydf,on='brand_id',how='left')
    
    mydf = df.groupby(['cat_id','spu_id']).size().reset_index()
    mydf = mydf.groupby(['cat_id']).size().reset_index()
    mydf = rename(mydf,0,'cat_spu_kinds_this_week')
    df = pd.merge(df,mydf,on='cat_id',how='left')
    
    mydf = df.groupby(['brand_id','cat_id']).size().reset_index()
    mydf = mydf.groupby(['cat_id']).size().reset_index()
    mydf = rename(mydf,0,'cat_brand_kinds_this_week')
    df = pd.merge(df,mydf,on='cat_id',how='left')
    
    return df 

def getWindowFeature(df,windows,num_weeks,config):
    flag = True
    for week in windows:
        print week
    
        temp = df[df['week']==week]
        if config['trick']:
            trick = getTrickFeature(temp)
            trick['week'] = week
            
        weeklst = range(week+1,week+num_weeks+1)
        temp2 = df[df.week.isin(weeklst)]
        
        if config['user']:
            goods_type = config['goods_type']
            user = temp[['uid']].drop_duplicates()
            for goods in goods_type:
                user1 = getUserFeature(temp2,user,goods,num_weeks)
                user = pd.merge(user,user1,on='uid',how='left')
            user['week'] = week
        if config['spu']:     
            spu = temp[['spu_id']].drop_duplicates()
            spu  = getGoodFeature(temp2,spu,'spu_id',num_weeks)
            spu['week'] = week
            
        if config['brand']:     
            brand = temp[['brand_id']].drop_duplicates()
            brand  = getGoodFeature(temp2,brand,'brand_id',num_weeks)
            brand['week'] = week
            
        if config['cat']:     
            cat = temp[['cat_id']].drop_duplicates()
            cat  = getGoodFeature(temp2,cat,'cat_id',num_weeks)
            cat['week'] = week
            
        if config['user2']:
            user2 = temp[['uid','spu_id','brand_id','cat_id']].drop_duplicates()
            user2  = getUserFeature2(temp2,user2,week,num_weeks)
            user2['week'] = week
        if flag:
            if config['user']:
                userFeature = user
            if config['spu']:
                spuFeature = spu
            if config['trick']:
                trickFeature = trick
            if config['brand']:
                brandFeature = brand
            if config['cat']:
                catFeature = cat               
            if config['user2']:
                userFeature2 = user2
            flag = False
        else:
            if config['user']:
                userFeature = userFeature.append(user)
            if config['spu']:
                spuFeature = spuFeature.append(spu)
            if config['trick']:
                trickFeature = trickFeature.append(trick)   
            if config['brand']:
                brandFeature = brandFeature.append(brand)   
            if config['cat']:
                catFeature = catFeature.append(cat)   
            if config['user2']:
                userFeature2 = userFeature2.append(user2)
    if config['user']:    
        userFeature.to_csv('../features/base/userFeatures_'+str(num_weeks)+'.csv',index=0)
        print 'user features writed'
    if config['user2']:    
        userFeature2.drop(['brand_id','cat_id'],axis=1).to_csv('../features/base/userFeatures2_'+str(num_weeks)+'.csv',index=0)
        print 'user features2 writed'
    if config['spu']:
        spuFeature.to_csv('../features/base/spuFeatures_'+str(num_weeks)+'.csv',index=0)
        print 'spu features writed'
    if config['trick']:
        trickFeature.drop(['label','brand_id','cat_id'],axis=1).to_csv('../features/base/trickFeatures.csv',index=0)
        print 'trick features writed'
    if config['brand']:
        brandFeature.to_csv('../features/base/brandFeatures_'+str(num_weeks)+'.csv',index=0)
        print 'brand features writed'
    if config['cat']:
        catFeature.to_csv('../features/base/catFeatures_'+str(num_weeks)+'.csv',index=0)
        print 'cat features writed'
    features = df[df.week.isin(windows)][['uid','spu_id','week','label','brand_id','cat_id']]
    if config['index']:
        features.to_csv('../features/base/IndexFeature.csv',index=0)
    #features = pd.merge(features,userFeature,on=['uid','week'],how='left')
    #features = pd.merge(features,spuFeature,on=['spu_id','week'],how='left')
    #features = pd.merge(features,trickFeature,on=['uid','spu_id','week'],how='left').fillna(0)
    return features

def getFeature(weeklst, task, sed):
	Log('...start read 1..')
	featureIndex = pd.read_csv('../features/base/IndexFeature.csv')
	Log('...start read 2..')
	userFeature9 = pd.read_csv('../features/base/userFeatures_9.csv')
	# userFeature8 = pd.read_csv('../features/base/userFeatures_8.csv')
	userFeature4 = pd.read_csv('../features/base/userFeatures_4.csv')
	userFeature1 = pd.read_csv('../features/base/userFeatures_1.csv')
	Log('...start read 3..')
	spuFeature9 = pd.read_csv('../features/base/spuFeatures_9.csv')
	# spuFeature8 = pd.read_csv('../features/base/spuFeatures_8.csv')
	spuFeature4 = pd.read_csv('../features/base/spuFeatures_4.csv')
	spuFeature1 = pd.read_csv('../features/base/spuFeatures_1.csv')
	
	Log('...start read 4..')
	brandFeature9 = pd.read_csv('../features/base/brandFeatures_9.csv')
	# brandFeature8 = pd.read_csv('../features/base/brandFeatures_8.csv')
	brandFeature4 = pd.read_csv('../features/base/brandFeatures_4.csv')
	brandFeature1 = pd.read_csv('../features/base/brandFeatures_1.csv')
	
	Log('...start read 5..')
	catFeature9 = pd.read_csv('../features/base/catFeatures_9.csv')
	# catFeature8 = pd.read_csv('../features/base/catFeatures_8.csv')
	catFeature4 = pd.read_csv('../features/base/catFeatures_4.csv')
	catFeature1 = pd.read_csv('../features/base/catFeatures_1.csv')
	
	Log('...start read 6..')
	userFeature29 = pd.read_csv('../features/base/userFeatures2_9.csv')
	# userFeature28 = pd.read_csv('../features/base/userFeatures2_8.csv')	
	userFeature24 = pd.read_csv('../features/base/userFeatures2_4.csv')
	userFeature21 = pd.read_csv('../features/base/userFeatures2_1.csv')
	
	Log('...start read 7..')
	trickFeature = pd.read_csv('../features/base/trickFeatures.csv')

	Log('...start merge 1..')
	df = pd.merge(featureIndex,userFeature9,on=['uid','week'],how='left')
	df = pd.merge(df,spuFeature9,on=['spu_id','week'],how='left')
	df = pd.merge(df,brandFeature9,on=['brand_id','week'],how='left')
	df = pd.merge(df,catFeature9,on=['cat_id','week'],how='left')
	df = pd.merge(df,userFeature29,on=['spu_id','week','uid'],how='left')
	df = pd.merge(df,trickFeature,on=['uid','spu_id','week'],how='left').fillna(0)
	Log('...start merge 2..')
	# df = pd.merge(df,userFeature8,on=['uid','week'],how='left')
	# df = pd.merge(df,spuFeature8,on=['spu_id','week'],how='left')
	# df = pd.merge(df,brandFeature8,on=['brand_id','week'],how='left')
	# df = pd.merge(df,catFeature8,on=['cat_id','week'],how='left')
	# df = pd.merge(df,userFeature28,on=['spu_id','week','uid'],how='left')

	df = pd.merge(df,userFeature4,on=['uid','week'],how='left')
	df = pd.merge(df,spuFeature4,on=['spu_id','week'],how='left')
	df = pd.merge(df,brandFeature4,on=['brand_id','week'],how='left')
	df = pd.merge(df,catFeature4,on=['cat_id','week'],how='left')
	df = pd.merge(df,userFeature24,on=['spu_id','week','uid'],how='left')
	Log('...start merge 3..')
	df = pd.merge(df,userFeature1,on=['uid','week'],how='left')
	df = pd.merge(df,spuFeature1,on=['spu_id','week'],how='left')
	df = pd.merge(df,brandFeature1,on=['brand_id','week'],how='left')
	df = pd.merge(df,catFeature1,on=['cat_id','week'],how='left')
	df = pd.merge(df,userFeature21,on=['spu_id','week','uid'],how='left')



	# weeklst = [1,2]
	trainF = df[df.week.isin(weeklst)].drop(['week','label','uid'],axis=1)
	trainL = df[df.week.isin(weeklst)].label.values
	testFeature = df[df.week==task].drop(['week','label','uid'],axis=1)

	# print trainFeature.shape,testFeature.shape
	start = 0
	for s in sed:
		Log(s)
		trainFeature, trainLabel = sampling(trainF,trainL,s)
		trainFeature['label'] = trainLabel
		Log('get feature over...')
		if task != -1:
			testLabel = df[df.week==task].label.values
			testFeature['label'] = testLabel

			trainFeature.to_csv('../features/localtrain/localtrain'+str(s)+'.csv', index = False)
			if start == 0:
				testFeature.to_csv('../features/localtest.csv', index = False)
		else:
			trainFeature.to_csv('../features/onlinetrain/onlinetrain'+str(s)+'.csv', index = False)
			if start == 0:
				testFeature.to_csv('../features/onlinetest.csv', index = False)
		start += 1

def readLocalOnlineFeature(sed):
	Log('read local feature...')
	train = pd.read_csv('../features/localtrain/localtrain'+str(sed)+'.csv', header = 0)
	test = pd.read_csv('../features/onlinetest.csv', header = 0)
	Log('read local feature over...')
	trainF = train.drop(['label'], axis = 1)
	testF = test[trainF.columns.values]
	trainLabel = train.label.values
	# testLabel = test.label.values
	Log(trainF.shape)
	Log(testF.shape)
	return trainF, trainLabel, testF
	
def readOnlineOnlineFeature(sed):
	Log('read online feature...')
	train = pd.read_csv('../features/onlinetrain/onlinetrain'+str(sed)+'.csv', header = 0)
	test = pd.read_csv('../features/onlinetest.csv', header = 0)
	Log('read online feature over...')
	trainF = train.drop(['label'], axis = 1)
	testF = test[trainF.columns.values]
	trainLabel = train.label.values
	return trainF, trainLabel, testF

def storeResult(preds,name,mode):
    result = pd.DataFrame(preds)
    result[0] = result[0].apply(lambda x:round(x,3))
    result.to_csv('../result/'+mode+'/'+name+'.txt',index=0,sep='\t',header = None)
    return result

def sampling(trainFeature,trainLabel, sed):
    n = len(trainLabel[trainLabel == 1])
    print n
    index = np.array(range(trainFeature.shape[0]))
    neg_mask = trainLabel==0
    neg = index[neg_mask]
    pos = index[~neg_mask]
    random.seed(sed)
    neg_sample = random.sample(neg, n)

    i = np.append(pos, neg_sample)
    i = random.sample(i,len(i))
    trainFeature = trainFeature.iloc[i,:]
    trainLabel = trainLabel[i]
    
    return trainFeature,trainLabel

def overSampling(trainFeature,trainLabel,rate = 1):
    n = int((len(trainLabel[trainLabel == 0]))/len(trainLabel[trainLabel == 1])/rate) -1
    print n
    pos_mask = trainLabel==1
    posFeature = trainFeature[pos_mask]
    posLabel = trainLabel[pos_mask]
    for i in range(n):
        trainFeature = trainFeature.append(posFeature)
        trainLabel = np.append(trainLabel,posLabel)
    trainFeature = trainFeature.reset_index(drop=True)
    # print trainFeature.shape
    return trainFeature,trainLabel

def blend(sedList, mode, name):
    start = 0
    for sed in sedList:
        r = pd.read_csv('../result/'+mode+'/final-'+mode+'-'+str(sed)+'.txt', sep = '\t', header = None)
        r.columns = [str(mode)+'r'+str(sed)]
        if start == 0:
            result = r
        else:
            result = pd.concat([result, r], axis = 1)
        start += 1

    print result.head()
    rr = result.mean(axis = 1).values
    storeResult(rr,name, 'blend')

def final_blend(eta):
    r1 = pd.read_csv('../result/blend/xgb-online-online-blend-50.txt', sep = '\t', header = None)
    r2 = pd.read_csv('../result/blend/lgb-online-online-blend-50.txt', sep = '\t', header = None)
    r3 = pd.read_csv('../result/blend/xgb-local-online-blend-50.txt', sep = '\t', header = None)
    r4 = pd.read_csv('../result/blend/lgb-local-online-blend-50.txt', sep = '\t', header = None)
    r1.columns = ['r1']
    r2.columns = ['r2']
    r3.columns = ['r3']
    r4.columns = ['r4']
    r = pd.concat([r1,r2,r3,r4], axis = 1)
    r['p'] = eta[0]*r['r1']+eta[1]*r['r2']+eta[2]*r['r3']+eta[3]*r['r4']
    storeResult(r['p'].values, 'final-result','blend')
