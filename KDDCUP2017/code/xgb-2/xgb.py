#coding:utf-8

import xgboost as xgb
import numpy as np
import pandas as pd
import math
import load

'''
def mapeobj(preds,dtrain):
    gaps = dtrain.get_label()
    
    #wei = dtrain.get_weight()
    grad = np.sign(preds-gaps)/gaps
    hess = 1/gaps
    
    t = []
    for x in list(abs(preds-gaps)):
        t.append(math.log(15.0+x, 15.0))
    t = np.array(t)
    grad = np.sign(preds-gaps)*t/gaps
    hess = 1/gaps**1/5

    grad[(gaps==0)] = 0
    hess[(gaps==0)] = 0
    return grad,hess  
'''


'''
def mapeobj(preds,dtrain):
    gaps = dtrain.get_label()
    #wei = dtrain.get_weight()
    t = []
    for x in list(abs(preds-gaps)):
        t.append(math.log(1.65+x, 1.65))    #1.65
    t = np.array(t)
    grad = np.sign(preds-gaps)*t/gaps
    hess = 1/gaps**1/5
    
    grad[(gaps==0)] = 0
    hess[(gaps==0)] = 0
    return grad,hess  
'''

base = np.e   #1.2
bot = 1 #1.3

def go(yt):
    y = list(yt)
    for i in range(len(y)):
        y[i] = math.log(y[i] + bot, base)
    return y/np.log(np.e)
    
def come(y):
    return np.power(base, y) - bot


def evalmape(preds, dtrain):
    '''
    labels = dtrain.get_label()     
    preds = come(preds) #tttt
    return 'mape', 1 -  np.abs((labels - preds) / labels).sum() / len(labels)
    '''
    
    gaps = dtrain.get_label()
    err = abs(gaps-preds)/gaps
    err[(gaps==0)] = 0
    err = np.mean(err)
    return 'mape-eval', err
    

def XGBoostSearch(X_train, y_train, X_test, y_test = None, flag = True, wei=None):
    cw = 0.0
    for t in y_train:
        cw += 1.0/t
    cw = len(y_train)/cw
            
    wei = []
    for t in y_train:
        wei.append(cw * 1.0/t)
    
    dtrain = xgb.DMatrix(X_train, label = y_train, missing = np.nan, weight = wei)
    
    #y = go(y_train)
    #dtrain = xgb.DMatrix(X_train, label = y, missing = np.nan)
    
    #线上提交
    random_seed = 128   #1225
    params={
    'booster':'gbtree', #  gbtree gblinear
    'objective': 'reg:linear',  #linear, logistic, gamma
    #'objective': 'reg:logistic',
    'early_stopping_rounds':50,
    #'eval_metric': 'mae',

    #'max_depth':14, #12
    'max_depth':18, #12
    'min_child_weight':1,   #20
    
    'subsample':0.9,    #0.9
    'colsample_bytree':0.9, #0.9
    
    
    'lambda':50,   #50
    'gamma':0.1,  #0.5

    
    'eta': 0.003,    #0.02 0.003
    #'eta': 0.8,    #0.02 0.003
    'seed':random_seed,
    'nthread':16,
    
    'stratified':True,
    'silent':1
    }
    
    #num_round = 300  #1000
    num_round = 10000  #1000
    
    dtest = xgb.DMatrix(X_test, label=y_test, missing = np.nan)
    evallist  = [(dtrain,'train'), (dtest,'test')]
    #bst = xgb.train( params, dtrain, num_round, evallist, obj=mapeobj, feval=evalmape)

    
    dep = [14]
    mcw = [1]
    smp = [0.85]
    lbd = [1, 3, 5, 10, 15]
    gma = [0.01, 0.1, 0.5, 1.0, 2.0]
    

    '''
    dep = [14]
    mcw = [1]
    smp = [0.85]
    lbd = [10]
    gma = [0.5]
    '''
    
    best_iter = 0
    best_sco = 0.0
    best_params = {}
    best_ntree = 0
    
    for di in dep:
        params['max_depth'] = di
        for mi in mcw:
            params['min_child_weight'] = mi
            for si in smp:
                params['subsample'] = si
                params['colsample_bytree'] = si
                for li in lbd:
                    params['lambda'] = li    
                    for gi in gma:
                        params['gamma'] = gi  

                        bst = xgb.train( params, dtrain, num_round, evallist, early_stopping_rounds=50, feval=evalmape)    
    
                        print '--------------------------------'
                        print 'this params:', params
                        print 'iter:', bst.best_iteration
                        print 'sco', 1.0 - bst.best_score
                        print 'tree', bst.best_ntree_limit 
                        
                        if bst.best_score > best_sco:
                            
                            file=open('search.parameter.tsv', 'a')
                            file.write('\n\n----------\n')
                            file.write('tree:'+str(bst.best_ntree_limit)+'\n')
                            file.write('sco:'+str(1.0-bst.best_score)+'\n')
                            for k, v in params.items():
                                file.write(str(k)+':' + str(v)+'\n')
                            file.close()
                            best_iter = bst.best_iteration
                            best_params = params
                            best_sco = bst.best_score
                            best_ntree = bst.best_ntree_limit 
                        
                        print '------------------------------'
                        print 'best params:', best_params
                        print 'best iter:', best_iter
                        print 'best sco:', 1.0 - best_sco
                        print 'best tree:', best_ntree


def XGBoost(X_train, y_train, X_test, y_test = None, flag = True, wei=None):
    #y = go(y_train)
    
    cw = 0.0
    for t in y_train:
        cw += 1.0/t
    cw = len(y_train)/cw
            
    wei = []
    for t in y_train:
        wei.append(cw * 1.0/t)
    
    dtrain = xgb.DMatrix(X_train, label = y_train, missing = -1, weight = wei)

    
    #dtrain = xgb.DMatrix(X_train, label = y, missing = np.nan, weight = wei)
    
    #print wei
    #train_Y = dtrain.get_label()
    
    '''
    random_seed = 128   #1225
    params={
    'booster':'gbtree', #  gbtree gblinear
    'objective': 'reg:linear',  #linear, logistic, gamma
    #'objective': 'reg:logistic',
    'early_stopping_rounds':100,
    #'eval_metric': 'mae',

    'max_depth':14, #8
    'min_child_weight':1,   #20
    
    'subsample':0.85,    #0.9
    'colsample_bytree':0.85, #0.9
    
    
    'lambda':10,   #50
    'gamma':0.5,  #0.5

    
    'eta': 0.003,    #0.2 0.5
    'seed':random_seed,
    'nthread':16,
    
    'stratified':True,
    'silent':1
    }
    
    num_round = 1249  #1000
    '''
    
    
    #线上提交
    random_seed = 2017   #1225   128
    params={
    #'gpu_id':1,
    #'updater':'grow_gpu',
            
    'booster':'gbtree', #  gbtree gblinear
    'objective': 'reg:linear',  #linear, logistic, gamma
    #'objective': 'reg:logistic',
    #'eval_metric': 'mae',

    #'max_depth':14, #12
    'max_depth':14, #14
    'min_child_weight':1,   #1
    
    'subsample':0.8,    #0.9
    'colsample_bytree':0.8, #0.9
    
    
    'lambda':35,   #35
    #'gamma':2.0, #0.5

    
    'eta': 0.003,    #0.02 0.003
    #'eta': 0.02,    #0.02 0.003
    #'eta': 0.006,    #0.02 0.003
    'seed':random_seed,
    'nthread':16,
    
    'stratified':True,
    'silent':1
    }
    
    #num_round = 300  #1000
    num_round = 2969 #2969
    
    
    
    if flag:
        dtest = xgb.DMatrix(X_test, label=y_test, missing = np.nan)
         
        evallist  = [(dtrain,'train'), (dtest,'test')]
        #bst = xgb.train( params, dtrain, num_round, evallist, obj=mapeobj, feval=evalmape)

        bst = xgb.train( params, dtrain, num_round, evallist, early_stopping_rounds=50, feval=evalmape)      
        
        print 'sco:', 1- bst.best_score
        print 'tree:',bst.best_ntree_limit 
        ypred = bst.predict(dtest)
        print ypred
        
        '''
        X = []
        for i in xrange(0, len(ypred)):
            x = []
            x.append(ypred[i])
            x.append(y_test[i])
            x.append(abs(y_test[i] - ypred[i])/y_test[i])
            X.append(x)
        df = pd.DataFrame(X, columns=['pred', 'y', 'mape'])
        #df = pd.DataFrame(ypred)
        df.to_csv('data/test.pred.y.mape.tsv', sep='\t', index=False, header=False)
        
        
        import tools
        tools.feature_important(bst, 'feat/feat.important.tsv')        
        #ypred = bst.predict(dtrain)
        #df = pd.DataFrame(ypred)
        #df.to_csv('data/train.tsv', sep='\t', index=False, header=False)
        #print ypred
        '''
    else:
        evallist  = [(dtrain,'train')]
        #bst = xgb.train( params, dtrain, num_round, evallist, obj=mapeobj, feval=evalmape)
        bst = xgb.train( params, dtrain, num_round, evallist, feval=evalmape)      
        
        dtest = xgb.DMatrix(X_test, missing = np.nan)
        ypred = bst.predict(dtest)
    
        return ypred
        #return come(ypred)


def test(filename):
    
    X_train, y_train, X_test, id_test = load.get_train_val()
    
    ypred = XGBoost(X_train, y_train, X_test, flag=False)
    for i in xrange(0, len(id_test)):
        id_test[i].append(ypred[i])
    
    df = pd.DataFrame(id_test, columns=['intersection_id','tollgate_id','time_window','avg_travel_time'])

    df.to_csv(filename+'.csv', sep=',', index=False)
    print df.avg_travel_time.mean()
    
    '''
    df = pd.read_csv('res/'+filename+'.csv', sep=',')
    dfr = pd.read_csv('data/submission_sample_travelTime.csv', sep=',')
    dfr.drop(['avg_travel_time'], axis=1, inplace=True)
    df = pd.merge(dfr, df, how='left', on=['intersection_id','tollgate_id','time_window'])
    #print df
    df.dropna(axis=0, how='any', inplace=True) #删除包含nan的行
    df.to_csv(filename+'.csv', sep=',', index=False)       
    '''

def train():
    X_train, y_train, X_test, y_test, w = load.get_train_data()
    
    XGBoost(X_train, y_train, X_test, y_test, wei=w)
    
    #XGBoostSearch(X_train, y_train, X_test, y_test, wei=w)
    
def main():
    #train()
    test('../../result/xgb-2')
    
if __name__ == "__main__":
    main()