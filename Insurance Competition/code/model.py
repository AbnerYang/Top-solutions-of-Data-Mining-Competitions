# -*- encoding:utf-8 -*-
import numpy as np 
import pandas as pd 
import copy
from frame import *
import xgboost as xgb
import lightgbm as lgb
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import KFold
import time as T

def Log(info):
    print T.strftime("%Y-%m-%d %H:%M:%S", T.localtime())+' '+str(info)

lowB = 0.24    
def calFscore(preds, origin):
    p = []
    num = 0
    for v in preds:
        if v > lowB:
            p.append(1)
        else:
            num += 1
            p.append(0)
    origin = list(origin)
    metric = pd.DataFrame({'predict':p,'true':origin})
    P = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.predict == 1)].shape[0]
    R = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.true == 1)].shape[0]
    return 2*P*R/(P + R)
    
def evalRmse(preds,dtrain):
    p = []
    num = 0
    for v in preds:
        if v > lowB:
            p.append(1)
        else:
            num += 1
            p.append(0)
    label = dtrain.get_label()
    metric = pd.DataFrame({'predict':p,'true':label})
    P = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.predict == 1)].shape[0]
    R = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.true == 1)].shape[0]
    return 'F1',2*P*R/(P + R)

def evalRmseLGB(preds,dtrain):
    p = []
    num = 0
    for v in preds:
        if v > lowB:
            p.append(1)
        else:
            num += 1
            p.append(0)
    label = dtrain.get_label()
    metric = pd.DataFrame({'predict':p,'true':label})
    P = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.predict == 1)].shape[0]
    R = float(metric[(metric.true == 1) & (metric.predict == 1)].shape[0])/metric[(metric.true == 1)].shape[0]
    return 'F1',2*P*R/(P + R), False


def get_weight(y):
    w = []
    for line in y:
        if line == 0:
            w.append(100)
        else:
            w.append(1)
    return w




class SKlearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, has_weight = False):
        paramss = copy.deepcopy(params)
        paramss['random_state'] = seed
        self.has_weight = has_weight
        try:
            self.clf = clf(**paramss)
        except Exception, e:
            self.clf = clf(**params)
            print "Exception"

    def train(self, x_train, y_train):
        if self.has_weight == False:
            return self.clf.fit(x_train,y_train)
        else:
            print 'has weight ...'
            return self.clf.fit(x_train,y_train,sample_weight = get_weight(y_train))

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

    def fit(self,x,y):
        if self.has_weight == False:
            return self.clf.fit(x,y)
        else:
            print 'has weight ...'
            return self.clf.fit(x,y,sample_weight = get_weight(y))

    def feature_importances(self):
        print(self.clf.feature_importances_)



class XgbWrapper(object):
    def __init__(self, seed=0, params=None, rounds = 100, has_weight = False):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = rounds
        self.has_weight = has_weight
        self.gbdt = None
        self.seed = seed

    def train(self, x_train, y_train):
        if self.has_weight:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight = weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist  = [(dtrain,'train')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval = 100, feval = evalRmse)
        # return self.gbdt

    def train_test_weight(self, x_train, y_train, x_test, y_test, weight):
        print 'self-define weight...'
        dtrain = xgb.DMatrix(x_train, label=y_train, weight = weight)
        dtest = xgb.DMatrix(x_test, label = y_test)
        watchlist  = [(dtrain,'train'),(dtest,'dtest')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval = 1, feval = evalRmse)
        # return self.gbdt

    def train_test(self, x_train, y_train, x_test, y_test):
        if self.has_weight == True:
            print 'has weight...'
            weight =  get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight = weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label = y_test)
        watchlist  = [(dtrain,'train'),(dtest,'dtest')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval = 1, feval = evalRmse)
        # return self.gbdt, self.gbdt.predict(dtest)
        # return self.gbdt

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def feature_importances(self):
        return self.gbdt.get_fscore()

    def default_cv(self, x_train, y_train, nfold=5):
        if self.has_weight:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight = weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)

        return xgb.cv(self.param, dtrain, self.nrounds, nfold = nfold, verbose_eval = 100, feval = evalRmse, stratified=True)
        
    def my_cv(self, trainFs, trainLs, trainIds, x_test, nfold=3):
        print('train on %d feature %d sample' % (trainFs[0].shape[1], trainFs[0].shape[0] + trainFs[1].shape[0]))
        x_train1, x_train2 = trainFs[0], trainFs[1]
        y_train1, y_train2 = trainLs[0], trainLs[1]
        id_train1, id_train2 = trainIds[0], trainIds[1]
        ntrain1, ntrain2, ntest = x_train1.shape[0], x_train2.shape[0], x_test.shape[0]
        kf1 = KFold(n=ntrain1, n_folds=nfold, shuffle=True, random_state=self.seed)
        
        oof_test = np.zeros((ntest,))
        oof_test_kf = np.zeros((ntest, nfold))
        FscoreList = []
        cumFscore = 0.0
        print('starting cv with %d loop and %d feature' % (nfold * nfold, x_train1.shape[1]))
        for i, (trIndex1, teIndex1) in enumerate(kf1):
            p1trI, p1trO, p1trid = x_train1[trIndex1], y_train1[trIndex1], id_train1[trIndex1]
            p1teI, p1teO, p1teid = x_train1[teIndex1], y_train1[teIndex1], id_train1[teIndex1]
            kf2 = KFold(n=ntrain2, n_folds=nfold, shuffle=True, random_state=self.seed + i ** i + np.random.randint(0, 200))
            for j, (trIndex2, teIndex2) in enumerate(kf2):
                if j != nfold - 1:
                    continue
                p2trI, p2trO, p2trid = x_train2[trIndex2], y_train2[trIndex2], id_train2[trIndex2]
                p2teI, p2teO, p2teid = x_train2[teIndex2], y_train2[teIndex2], id_train2[teIndex2]
                
                trI, trO, trid = np.concatenate([p1trI, p2trI]), np.concatenate([p1trO, p2trO]), np.concatenate([p1trid, p2trid])
                teI, teO, teid = np.concatenate([p1teI, p2teI]), np.concatenate([p1teO, p2teO]), np.concatenate([p1teid, p2teid])
                Log('### Loop Start ###')
                print('cv-trainning loop: %d' % (nfold * i + j + 1))
                self.train(trI, trO)
                print('cv-predicting loop: %d' % (nfold * i + j + 1))
                pred, testP = self.predict(teI), self.predict(x_test.values)
                testP = np.array(testP).reshape(1, -1)
                oof_test_kf[:, i] = testP
                FscoreList.append(calFscore(pred, teO))
                testdf = pd.DataFrame(data=[[pred[i], teO[i]] for i in range(pred.shape[0])])
                curr = T.strftime('%d-%H-%M', T.localtime())
                cumFscore = cumFscore + FscoreList[-1]
                print('cv-result: Fold%d-Fscore[%.5f]' % (nfold * i + j + 1, FscoreList[-1]))
                Log('### Loop End ###')
        oof_test = np.matmul(oof_test_kf, np.array(FscoreList).reshape(-1, 1)) / np.sum(FscoreList)
        print('###############################')
        print('cv-result: mean-Fscore[%.5f]' % (cumFscore / float(nfold)))
        return cumFscore / float(nfold), oof_test


class LgbWrapper(object):
    def __init__(self, seed=0, params=None, rounds = 100, has_weight = False):
        self.param = params
        self.param['seed'] = seed
        self.param['silent']
        self.nrounds = rounds
        self.has_weight = has_weight
        self.seed = seed

    def train(self, x_train, y_train):
        if self.has_weight == True:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = lgb.Dataset(x_train, label=y_train, weight = weight)
        else:
            dtrain = lgb.Dataset(x_train, label=y_train)
        self.gbdt = lgb.train(self.param, dtrain, num_boost_round = self.nrounds, valid_sets = [dtrain], feval=evalRmseLGB, verbose_eval = 100)


    def train_test(self, x_train, y_train, x_test, y_test):
        if self.has_weight:
            print 'has weight...'
            weight =  get_weight(y_train)
            dtrain = lgb.Dataset(x_train, label=y_train, weight = weight)
        else:
            dtrain = lgb.Dataset(x_train, label=y_train)
        dtest = lgb.Dataset(x_test, label = y_test)
        self.gbdt = lgb.train(self.param, dtrain, self.nrounds, valid_sets = [dtest] , feval=evalRmseLGB, verbose_eval = 100)


    def predict(self, x):
        return self.gbdt.predict(x)

    def feature_importances(self):
        return self.gbdt.get_fscore()

    def default_cv(self, x_train, y_train, nfold=5):
        if self.has_weight == True:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = lgb.Dataset(x_train, label=y_train, weight = weight)
        else:
            dtrain = lgb.Dataset(x_train, label=y_train)

        return lgb.cv(self.param, dtrain, self.nrounds, nfold = nfold, feval=evalRmseLGB, verbose_eval = 200)
        
    def my_cv(self, trainFs, trainLs, trainIds, x_test, nfold=3):
        x_train1, x_train2 = trainFs[0], trainFs[1]
        y_train1, y_train2 = trainLs[0], trainLs[1]
        ntrain1, ntrain2, ntest = x_train1.shape[0], x_train2.shape[0], x_test.shape[0]
        kf1 = KFold(n=ntrain1, n_folds=nfold, shuffle=True, random_state=self.seed)
        
        oof_test = np.zeros((ntest,))
        oof_test_kf = np.zeros((ntest, nfold))
        FscoreList = []
        cumFscore = 0.0
        print('starting cv with %d loop' % (nfold * nfold))
        for i, (trIndex1, teIndex1) in enumerate(kf1):
            p1trI, p1trO = x_train1[trIndex1], y_train1[trIndex1]
            p1teI, p1teO = x_train1[teIndex1], y_train1[teIndex1]
            kf2 = KFold(n=ntrain2, n_folds=nfold, shuffle=True, random_state=self.seed + i ** i + np.random.randint(0, 200))
            for j, (trIndex2, teIndex2) in enumerate(kf2):
                if j != nfold - 1:
                    continue
                p2trI, p2trO = x_train2[trIndex2], y_train2[trIndex2]
                p2teI, p2teO = x_train2[teIndex2], y_train2[teIndex2]
                
                trI, trO = np.concatenate([p1trI, p2trI]), np.concatenate([p1trO, p2trO])
                teI, teO = np.concatenate([p1teI, p2teI]), np.concatenate([p1teO, p2teO])
                Log('### Loop Start ###')
                print('cv-trainning loop: %d' % (nfold * i + j + 1))
                self.train(trI, trO)
                print('cv-predicting loop: %d' % (nfold * i + j + 1))
                pred, testP = self.predict(teI), self.predict(x_test.values)
                testP = np.array(testP).reshape(1, -1)
                oof_test_kf[:, i] = testP
                FscoreList.append(calFscore(pred, teO))
                cumFscore = cumFscore + FscoreList[-1]
                print('cv-result: Fold%d-Fscore[%.5f]' % (nfold * i + j + 1, FscoreList[-1]))
                Log('### Loop End ###')
        oof_test = np.matmul(oof_test_kf, np.array(FscoreList).reshape(-1, 1)) / np.sum(FscoreList)
        print('###############################')
        print('cv-result: mean-Fscore[%.5f]' % (cumFscore / float(nfold)))
        return cumFscore / float(nfold), oof_test

