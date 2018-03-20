#-*- encoding:utf-8 -*-
import numpy as np 
import pandas as pd
import time as T
import copy
import scipy as sp
import os
import warnings
import random
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
import sys
from nmf import nmf
reload(sys)
sys.setdefaultencoding('utf-8')

def Log(info):
    print T.strftime("%Y-%m-%d %H:%M:%S", T.localtime())+' '+str(info)


def getUseColumn(data):
    d = copy.deepcopy(data)
    k = d.var()
    # print k
    col1 = k[(k != np.nan) & (k != 0)].index.values
    return col1

def getDetailFeature(feature, data):
    """
        对fee_detail.csv，从【三目统计项目、三目服务项目名称、拒付原因编码、单价、数量】进行特征提取
        由于前三个字段均为类别变量，因此处理方式：
        首先，都是进行one-hot编码
        随后，分别计算每个人的这些编码字段（类别变量）的统计信息【mean, sum, std】
        接着，产生一些新的连续型变量，并对所有连续变量产生统计信息【mean, sum, median, min, max, std】
    """
    df = copy.deepcopy(data)
    df['费用发生时间'] = df['费用发生时间'].map(pd.Timestamp)
    
    df['月份'] = df['费用发生时间'].map(lambda ts: ts.month)
    df['星期'] = df['费用发生时间'].map(lambda x: x.isoweekday())
    
    
    colD = []
    for i in [1, 2, 3, 4, 5, 6, 7]:
        df['星期' + str(i)] = [[0, 1][value == i] for value in df['星期'].values]
        colD.append('星期' + str(i))
        
    
    colM = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        df['月份_' + str(i)] = [[0, 1][value == i] for value in df['月份'].values]
        colM.append('月份_' + str(i))
        
    case1 = df['三目服务项目名称'].replace(np.nan, u' ').values
    col1 = []
    for i in [1,2,3,4,5,6,7,9]:
        df['三目统计项目_'+str(i)] = [[0,1][value == i] for value in df['三目统计项目'].values]
        col1.append('三目统计项目_'+str(i))
        
    col2 = []
    for i in [1,2]:
        df['拒付原因编码_'+str(i)] = [[0,1][value == i] for value in df['拒付原因编码'].values]
        col2.append('拒付原因编码_'+str(i))
    df['是否拒付'] = df['拒付原因编码_1'] + df['拒付原因编码_2']
    col2.append('是否拒付')
    case1 = df['三目服务项目名称'].replace(np.nan, u' ').values
    col3 = []
    k = ['普通门诊挂号费','挂号','门特','三级医院','一级医院','二级医院',\
    '副主任医师','主任医师','专家号', '公费']
    for i in k:
        df['三目服务项目名称_'+i] = [[0,1][i in value] for value in case1]
        col3.append('三目服务项目名称_'+i)

    Log('get dummy feature...')
    col_d = col1+col2+col3+colD+colM
    feature_dum = df.groupby(['uid'])[col_d].agg([np.mean, np.sum, np.std])
    feature_dum = feature_dum.reset_index()
    kind_l = ['sum', 'mean', 'std']
    for kind in kind_l:
        feature_dum['月份_'+kind+'_std'] = feature_dum[colM].stack(level=0)[kind].unstack(level=1).std(axis=1)
    feature = pd.merge(feature, feature_dum, on = 'uid', how = 'left')

    Log('get continuous feature...')
    df['三目统计项目_sum'] = df[col1].sum(axis = 1)
    df['拒付原因编码_sum'] = df[col2].sum(axis = 1)
    df['三目服务项目名称_sum'] = df[col3].sum(axis = 1)
    df['工作日_sum'] = df[colD[:5]].sum(axis = 1)
    df['周末_sum'] = df[colD[5:]].sum(axis = 1)
    col_c_1 = []
    k = ['(',' ',',',';','*','-','；','，','（']
    for i in k:
        df['case_三目服务项目名称_'+i] = [len(value.split(i)) for value in case1]
        col_c_1.append('case_三目服务项目名称_'+i)


    col_c = ['单价','三目统计项目_sum','数量','拒付原因编码_sum','三目服务项目名称_sum', '工作日_sum', '周末_sum']
    con_func = [np.mean, np.sum, np.median, np.min, np.max, np.std]
    col_c = col_c + col_c_1
    feature_con = df.groupby(['uid'])[col_c].agg(con_func)
    feature_con = feature_con.reset_index()
    feature = pd.merge(feature, feature_con, on = 'uid', how = 'left')
    
    feature_date = df.groupby(['uid'])['费用发生时间'].apply(lambda g: g.sort_values().diff().map(lambda d: d.days).agg(con_func)).unstack()
    feature_date.index.name = 'uid'
    feature_date.columns = ['费用发生时间' + postfix for postfix in feature_date.columns.values]
    feature_date = feature_date.reset_index()
    feature = pd.merge(feature, feature_date, on='uid', how='left')
    
    return feature


def getMainFeature(feature, data):
    """
        对df_train(test).csv进行特征提取
    """
    Log('...get main feature...')
    df = copy.deepcopy(data)
    # 诊断结果病种提取
    case = df['出院诊断病种名称'].replace(np.nan, u' ').values
    df['case_length'] = [len(value) for value in case]
    df['心脏病'] = [[0,10]['心脏病' in value] for value in case]
    df['肺心病'] = [[0,4]['肺心病' in value] for value in case]
    df['高血压'] = [[0,17]['高血压' in value] for value in case]
    df['冠心病'] = [[0,10]['冠心病' in value] for value in case]
    df['挂号'] = [[0,7]['挂号' in value] for value in case]
    df['门特挂号'] = [[0,13]['门特挂号' in value] for value in case]
    df['糖尿病'] = [[0,45]['糖尿病' in value] for value in case]
    df['尿毒症'] = [[0,5]['尿毒症' in value] for value in case]
    df['偏瘫'] = [[0,14]['偏瘫' in value] for value in case]
    df['精神病'] = [[0,2]['精神病' in value] for value in case]
    
    df['是否残疾军人'] = 0
    df.loc[(~df['残疾军人医疗补助基金支付金额'].isnull()) | (df['残疾军人医疗补助基金支付金额'] != 0), '是否残疾军人'] = 1
    df['是否城乡救助'] = 0
    df.loc[(~df['城乡救助补助金额'].isnull()) | (df['城乡救助补助金额'] != 0), '是否城乡救助'] = 6
    df['是否公务员'] = 0
    df.loc[(~df['公务员医疗补助基金支付金额'].isnull()) | (df['公务员医疗补助基金支付金额'] != 0), '是否公务员'] = 4
    df['是否自负'] = 0
    df.loc[(~df['起付标准以上自负比例金额'].isnull()) | (df['起付标准以上自负比例金额'] != 0), '是否自负'] = 1
    df['是否民政救助'] = 0
    df.loc[(~df['民政救助补助金额'].isnull()) | (df['民政救助补助金额'] != 0), '是否民政救助'] = 4
    df['是否城乡优抚'] = 0
    df.loc[(~df['城乡优抚补助金额'].isnull()) | (df['城乡优抚补助金额'] != 0), '是否城乡优抚'] = 3
    # 诊断结果分隔符数量提取
    df['case_has_type1'] = [len(value.split(','))-1 for value in case]
    df['case_has_type2'] = [len(value.split('，'))-1 for value in case]
    df['case_has_type3'] = [len(value.split(';'))-1 for value in case]
    df['case_has_type4'] = [len(value.split('；'))-1 for value in case]
    df['case_has_type5'] = [len(value.split('（'))-1 for value in case]
    df['case_has_type6'] = [len(value.split(' '))-1 for value in case]
    # 统计信息
    case_col1 = ['心脏病','肺心病','高血压','冠心病','挂号','门特挂号','糖尿病','尿毒症','偏瘫','精神病']
    case_col_sp = ['是否残疾军人', '是否城乡救助', '是否公务员', '是否自负', '是否民政救助', '是否城乡优抚']
    df['病_sum'] = df[case_col1].sum(axis = 1)
    df['special_sum'] = df[case_col_sp].sum(axis=1)
    case_col2 = ['case_length','case_has_type1','case_has_type2','case_has_type3',\
    'case_has_type4','case_has_type5','case_has_type6','病_sum', 'special_sum']
    
    money = []
    no_money = []
    for value in df.columns.values:
        if '金额' in value:
            money.append(value)
        else:
            no_money.append(value)
    # 非负矩阵分解，得到用户矩阵和项目费用矩阵，将用户矩阵作为新的特征 (20维)     
    money_sum_df, money_mean_df = df.groupby(['uid'])[money].apply(np.sum).fillna(0), df.groupby(['uid'])[money].apply(np.mean).fillna(0)
    matdfs = {'MoneySum': money_sum_df, 'MoneyMean': money_mean_df}
    matvs = {k: item.copy().values for k, item in matdfs.items()}
    factor_num = 20
    for k in matdfs:
        matdf, matv = matdfs[k], matvs[k]
        shape = matv.shape
        np.random.seed(20)
        initW, initH = np.random.rand(shape[0], factor_num), np.random.rand(factor_num, shape[1])
        outW, outH = nmf(matv, initW, initH, 0.0001, 5555555, 250)
        feature_factor = pd.DataFrame(outW, index=pd.Index(data=matdf.index.values, copy=True, name='uid'), 
                                      columns=[k + '_factor_' + str(i) for i in range(factor_num)]).reset_index()
        print(feature_factor.max(axis=0))
        feature = pd.merge(feature, feature_factor, on='uid', how='left').fillna(0)
    # 获得用户记录条数
    Log('get record number feature...')
    feature_rec = df.groupby(['uid'])['顺序号'].count()
    feature_rec = feature_rec.rename('record_num')
    feature_rec = feature_rec.reset_index()
    feature = pd.merge(feature, feature_rec, on = 'uid', how = 'left')
    # 获得用户去的不同医院数量
    Log('get category feature...')
    feature_cat = df.groupby(['uid'])['医院编码'].nunique()
    feature_cat = feature_cat.rename('hospital_num')
    feature_cat = feature_cat.reset_index()
    feature = pd.merge(feature, feature_cat, on = 'uid', how = 'left')
    # 对连续型变量产生统计信息【mean, sum, median, min, max, std】
    Log('get continuous feature...')
    col_con = money + case_col2
    # col_con.append('医疗救助医院申请')
    # ,'医疗救助医院申请'
    feature_con = df.groupby(['uid'])[col_con].agg([np.mean, np.sum, np.median, np.min, np.max, np.std])
    feature_con = feature_con.reset_index()
    feature = pd.merge(feature, feature_con, on = 'uid', how = 'left')

    # 对类别变量产生统计信息【mean, sum, std】
    Log('get dummy feature...')
    feature_dum = df.groupby(['uid'])[case_col1 + case_col_sp].agg([np.mean, np.sum, np.std])
    feature_dum = feature_dum.reset_index()
    feature = pd.merge(feature, feature_dum, on = 'uid', how = 'left')
    # 非负矩阵分解，得到用户矩阵和医院矩阵，将用户矩阵作为新的特征 (15维)     
    factor_num = 15
    PH_Mat = df.groupby(['uid', '医院编码']).size().unstack().fillna(0)
    PH_MatV = PH_Mat.copy().values
    shape = PH_MatV.shape
    np.random.seed(20)
    initW, initH = np.random.rand(shape[0], factor_num), np.random.rand(factor_num, shape[1])
    outW, outH = nmf(PH_MatV, initW, initH, 0.0001, 5555555, 200)
    feature_factor = pd.DataFrame(outW, index=pd.Index(data=PH_Mat.index.values, copy=True, name='uid'), 
                                  columns=['Hostpital_factor_' + str(i) for i in range(factor_num)]).reset_index()
    feature = pd.merge(feature, feature_factor, on='uid', how='left').fillna(0)
    return feature


def getFeature(mode):
    """
        利用data文件夹中原始数据，调用特征提取函数，产生用于模型训练和预测的数据文件，并保存于feature文件夹
    """
    Log('..read record info...')
    df = pd.read_csv('../data/df_'+mode+'.csv', header = 0)
    df['uid'] = df['个人编码']
    df = df.drop(['个人编码'], axis = 1)

    Log('..read id info...')
    df_id = pd.read_csv('../data/df_id_'+mode+'.csv', header = None)
    
    if mode == 'train':
        df_id.columns = ['uid','label']
    else:
        df_id.columns = ['uid']
    
    feature = copy.deepcopy(df_id)
    Log('..read detail info...')
    df_detail = pd.read_csv('../data/fee_detail.csv', header = 0)
    detail_id_list = df['顺序号'].values
    Log(df_detail.shape)
    df_detail = df_detail[df_detail['顺序号'].isin(detail_id_list)]
    Log(df_detail.shape)

    df_detail = pd.merge(df_detail, df[['uid','顺序号']], on = '顺序号', how = 'left')
    Log('...get feature detail...')

    feature = getDetailFeature(feature,df_detail)
    feature = getMainFeature(feature,df)

    feature.to_csv('../feature/'+mode+'.csv', index = False, encoding="utf_8_sig")
    
    

    


def readFeature(Mode='P', idfile='../data/obviousId.csv'):
    """
        用于从feature文件夹中读入用于模型训练和预测的数据文件
    """
    paths = dict(train='../feature/train.csv', test='../feature/test.csv')
    for p in paths:
        if not os.path.exists('../feature'):
            os.makedirs('../feature')
        if not os.path.exists(paths[p]):
            getFeature(p)
    train = pd.read_csv(paths['train'], header = 0)
    test = pd.read_csv(paths['test'], header = 0)
    if Mode == 'R':
        removeIds = pd.read_csv(idfile, header=None).iloc[:, 0].values
        index = train.uid.map(lambda x: x not in removeIds)
        train = train.loc[index, :].copy()
    trainL = [train.loc[train['label'] == 0, :].copy(), train.loc[train['label'] == 1, :].copy()]
    
    trainLabel = train.label.values
    Log(train.shape)
    Log(test.shape)
    T.sleep(10)
    trainFeature = train.drop(['uid','label'], axis = 1)
    # 去除一些比较取值分布异常的字段
    trainFeature = trainFeature[getUseColumn(trainFeature)]
    testFeature = test[trainFeature.columns.values]
    trainLabel = train.label.values
    
    trainFs, trainLs, trainIds = [], [], []
    for L in trainL:
        trainFs.append(L[trainFeature.columns.values].values)
        trainLs.append(L.label.values)
        trainIds.append(L.uid.values)
        
    test.uid = test.uid.map(str)
    testIndex = test[['uid']]
    Log(trainFeature.shape)
    Log(testFeature.shape)
    
    return trainFeature, trainLabel, testFeature, testIndex, trainFs, trainLs, trainIds

def store(pred, testIndex, name, Mode='regular'):
    """
        输出模型预测结果信息，保存模型结果在(name).csv文件中
    """
    prefix = '../result/'
    if not os.path.exists('../result'):
        os.makedirs('../result')
    if Mode != 'regular':
        prefix = '../%s/' % Mode
    testIndex['predict'] = pred
    testIndex2 = testIndex.sort_values(by = 'predict', ascending = False)
    top200Prob, trustedSize = testIndex2.predict.values[200], testIndex2[testIndex2.predict > 0.1855].shape[0]
    Log(testIndex2.predict.values[200])
    Log(testIndex2[testIndex2.predict > 0.1855].shape[0])
    StoreInfo = '(%.5f,%d)' % (top200Prob, trustedSize)
    name = name + StoreInfo
    testIndex.to_csv(prefix+name+'(P).csv', header=False, index=False)
    testIndex.loc[testIndex.predict > 0.1855, 'predict'] = 1
    testIndex.loc[testIndex.predict <= 0.1855, 'predict'] = 0
    testIndex.predict = testIndex.predict.astype(int)
    testIndex.to_csv(prefix+name+'(L).csv', header = False, index = False)


if __name__ == '__main__':
    getFeature('train')
    getFeature('test')