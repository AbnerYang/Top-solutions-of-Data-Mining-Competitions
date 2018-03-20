
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
import pandas as pd


#标准化
def normalization(df):  
    X = df.values
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    print "preprocessing done!"
    df = pd.DataFrame(X, columns = df.columns) 
    return df


#加载特征
def loadData(path):
    train_x = pd.read_csv(path, sep=',')
    
    uid = train_x[['V1']]
    
    train_x.drop(['V1'], axis=1, inplace=True)
    train_y = pd.read_csv("data/idx.target.csv", sep=',')
    train_y.drop(['Idx'], axis=1, inplace=True)
    
    return train_x, train_y, uid

#cv
def fpreproc(dtrain, dtest, param):
    labels = dtrain.get_label()
    rat = float(np.sum(labels == 0)) / np.sum(labels==1)
    param['scale_pos_weight'] = rat
    return (dtrain, dtest, param)

#xgboost
def XGBoost(feat_name, train_X, train_Y, X_test):
    
    dtrain = xgb.DMatrix(train_X, label = train_Y, missing = -1, feature_names = feat_name)
    random_seed = 1288   #27
    params={
    'scale_pos_weight': float(np.sum(train_Y == 0)) / np.sum(train_Y==1),
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'stratified':True,

    'max_depth':3,
    'min_child_weight':4,
    
    'subsample':0.8,
    'colsample_bytree':0.8,
    
    'lambda':300,   #550
    'gamma':0.65,
    
    'eta': 0.02,
    'seed':random_seed,
    'nthread':8,
    
    'silent':1
    }
    
    num_round = 1350
    
    
    dtest= xgb.DMatrix(X_test, missing = np.nan, feature_names = feat_name)
    #评估训练集
    evallist  = [(dtrain,'train')]
    bst = xgb.train( params, dtrain, num_round, evallist)

    #预测
    ypred = bst.predict(dtest)
        
    return list(ypred)

#交叉验证， 默认5折交叉
def XGBoostCV(feat_name, X, y, nfolds = 5):
    from sklearn.cross_validation import KFold
    kf = KFold(len(y), n_folds=nfolds, shuffle = True, random_state = 345)
    
    scos = []
    k = 1
    for train_index, test_index in kf:
#        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        ans = XGBoost(feat_name, X_train, y_train, X_test, k)
        k += 1
        
        auc = score(y_test, ans)
        scos.append(auc)
        print auc
        
        del X_train, X_test, y_train, y_test
    print "\n\n",scos
    avg = 0.0
    for x in scos:
        avg += x
    print "avg auc:", avg/len(scos)    

#计算AUC
def score(label, pred): 
    neg_lis = []
    pos_lis = []
    for i in xrange(0, len(label)):
        if label[i] == 1:
            pos_lis.append(pred[i])
        else:
            neg_lis.append(pred[i])
            
    print "all:",len(label), "pos:",len(pos_lis), "neg:", len(neg_lis)
    neg_lis.sort()
    
    t = 0.0
    for i in xrange(0, len(pos_lis)):
        for j in xrange(0, len(neg_lis)):
            if pos_lis[i] > neg_lis[j]:
                t += 1
            elif pos_lis[i] == neg_lis[j]:
                t += 0.5
            else:
                break
    cnt = len(pos_lis)*len(neg_lis)
    print t, cnt
    return t/cnt

#
def SearchParameters(train_X, train_Y, F = 10):
    dtrain = xgb.DMatrix(train_X, label = train_Y, missing = -1)

    random_seed = 1288
    params={
    'scale_pos_weight': float(np.sum(train_Y == 0)) / np.sum(train_Y==1),
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'stratified':True,

    'max_depth':3,
    'min_child_weight':4,
    
    'subsample':0.8,
    'colsample_bytree':0.8,
    
    'lambda':300,   #550
    'gamma':0.65,
    
    'eta': 0.02,
    'seed':random_seed,
    'nthread':8,
    
    'silent':1
    }
    
    num_round = 3500   #200
    
    
    max_depth = [3,4,5]
    min_child_weight = [4,5,6] 
    
    t1 = 0
    t2 = 0
    t3 = 0
    max_auc = 0
    for a in max_depth:
        for b in min_child_weight:
                params['max_depth'] = a
                params['min_child_weight'] = b
                
                sco = xgb.cv(params, dtrain, num_round, nfold=F, metrics={'auc'}, verbose_eval = True, seed = 0, fpreproc = fpreproc)                
                mean_auc = sco['test-auc-mean']
                good_auc = np.array(mean_auc).max()  #最大的AUC
                
                niter = 0
                for i in xrange(0, len(mean_auc)):  #找到最大的迭代轮数
                    if mean_auc[i] == good_auc:
                        niter = i
                
                if good_auc > max_auc:
                    t1 = a
                    t2 = b
                    t3 = niter
                    max_auc = good_auc
                print "A:", good_auc, a, b, niter
                print "B:", max_auc, t1, t2, t3
                
    print "last_ auc: ", max_auc, t1, t2, t3


      

def main():
    train_x, train_y, uid_train = loadData("data/last_feature_3952/train.3952.csv")
    test_x, test_y, uid_test = loadData("data/last_feature_3952/test.3952.csv")    
    
    flag = 1    #flag=1 -> 训练3w测试2w,   flag=0 -> 训练8w测试1w
    
    
    if flag == 1:
        test_x = train_x.loc[30000:49998,:] #取2w测试数据
        uid_test = uid_train.loc[30000:49998,:]
        
        train_x = train_x.loc[:29999,:] #取3w训练数据
        train_y = train_y.loc[:29999,:]
    
        ##############################################
        df_nor = pd.concat([test_x, train_x], axis=0)
        df_nor = normalization(df_nor)
        
        test_x = df_nor.loc[:19998,:]
        train_x = df_nor.loc[19999:,:]
        
        
        ans = XGBoost(list(train_x.columns), train_x.values, train_y.values, test_x.values, 0)
        df = pd.DataFrame({"Idx":uid_test['V1'], "score":ans})
        df.to_csv("top/xgb.4-18.3952.round1350.eta0.02.train3w.test2w.csv", index=False)
    else:
        df_nor = pd.concat([test_x, train_x], axis=0)
        df_nor = normalization(df_nor)
        
        test_x = df_nor.loc[:9999,:]
        train_x = df_nor.loc[10000:,:]
        
        ans = XGBoost(list(train_x.columns), train_x.values, train_y.values, test_x.values, 0)
        df = pd.DataFrame({"Idx":uid_test['V1'], "score":ans})
        df.to_csv("ans/xgb.4-18.3952.round2280.eta0.02.train8w.test1w.csv", index=False)

    
if __name__ == "__main__":
    main()






