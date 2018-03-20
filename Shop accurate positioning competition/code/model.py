# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:37:20 2018

@author: 1
"""
import numpy as np
import pandas as pd
import time
import xgboost as xgb



class MLframe(object):
    """docstring for MLframe"""

    def __init__(self, train_x, train_y, test_x, seed=0):
        self.seed = seed
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def stacking(self, clf, nfold):
        oof_train = np.zeros([self.train_x.shape[0], 1])
        oof_test = np.zeros([self.test_x.shape[0], 1])
        oof_test_skf = np.empty([self.test_x.shape[0], nfold])
        skf = list(KFold(n_splits=nfold, random_state=self.seed).split(self.train_x, self.train_y))
        for i, (train_index, test_index) in enumerate(skf):
            LogInfo("--fold:" + str(i))
            x_tr = self.train_x[train_index, :]
            y_tr = self.train_y[train_index]
            x_te = self.train_x[test_index, :]
            print x_tr.shape, x_te.shape

            clf.train(x_tr, y_tr)

            oof_train[test_index, 0] = clf.predict(x_te)
            oof_test_skf[:, i] = clf.predict(self.test_x)

        evalu = evaluate(oof_train[:, 0], self.train_y)
        print evalu.mape()
        oof_test[:] = oof_test_skf.mean(axis=1).reshape(self.test_x.shape[0], 1)
        return oof_train, oof_test

    def train_test(self, clf, test_y, isxgb=False):
        if isxgb == True:
            clf.train_test(self.train_x, self.train_y, self.test_x, test_y)
        else:
            clf.train(self.train_x, self.train_y)
            predict = clf.predict(self.test_x)
            evalu = evaluate(predict, test_y)
            print evalu.mape()

    def train_predict(self, clf):
        clf.train(self.train_x, self.train_y)
        predict = clf.predict(self.test_x)
        return predict

    def get_importance(self, clf, featureName):
        print featureName[0:]
        clf.train(self.train_x, self.train_y)
        imp_f = clf.feature_importances()
        df = pd.DataFrame({'feature': [featureName[int(key[1:])] for key, value in imp_f.items()],
                           'fscore': [value for key, value in imp_f.items()]})
        # mean_imp = np.mean(df['fscore'])
        # top_f = df.loc[df['fscore'] >= mean_imp]
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        df = df.sort_values(by='fscore', ascending=False)
        print df


class evaluate(object):
    """docstring for evaluate"""

    def __init__(self, predict, truth):
        '''
        predict:row_id, shop_id, probability
        truth: row_id ,shop_id
        '''
        self.predict = predict
        self.true = truth

    def selfDefinePrecision(self):
        '''
        '''
        pass


class XgbWrapper(object):
    def __init__(self, seed=0, params=None, rounds=100, has_weight=False):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = rounds
        self.has_weight = has_weight

    def train(self, x_train, y_train):
        if self.has_weight == True:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        watchlist = [(dtrain, 'train')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval=1)
        # return self.gbdt

    def train_test(self, x_train, y_train, x_test, y_test):
        if self.has_weight == True:
            print 'has weight...'
            weight = get_weight(y_train)
            # print weight
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        watchlist = [(dtrain, 'train'), (dtest, 'dtest')]
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds, watchlist, verbose_eval=1)
        # return self.gbdt

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def feature_importances(self):
        return self.gbdt.get_fscore()

    def default_cv(self, x_train, y_train):
        if self.has_weight == True:
            print 'has weight...'
            weight = get_weight(y_train)
            dtrain = xgb.DMatrix(x_train, label=y_train, weight=weight)
        else:
            dtrain = xgb.DMatrix(x_train, label=y_train)
        xgb.cv(self.param, dtrain, self.nrounds, verbose_eval=20)