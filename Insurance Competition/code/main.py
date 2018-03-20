# -*- enconding:utf-8 -*-
import numpy as np 
import pandas as pd 
from frame import *
from model import *
from feature import *
# from base_feature import *
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.neural_network import MLPRegressor
import copy

import argparse

parser = argparse.ArgumentParser(description='ML Model')
parser.add_argument('--FeatureFold', default=3, type=int, metavar='N',
                    help='number of data FeatureFold (default: 3)')
parser.add_argument('--FeatureLoop', default=10, type=int, metavar='N',
                    help='number of FeatureLoop (default: 10)')
parser.add_argument('--BaggingLoop', default=10, type=int, metavar='N',
                    help='number of BaggingLoop (default: 10)')
parser.add_argument('--CVFold', default=3, type=int, metavar='N',
                    help='number of CVFold (default: 3)')
parser.add_argument('--ARCH', default='xgb', type=str, metavar='S',
                    help='use which(xgb/lgb...) model (default: xgb)')
parser.add_argument('--Part', default='More', type=str, metavar='S',
                    help='use which (More/Less...) part of feature (default: More)')
parser.add_argument('--Remove', default='No', type=str, metavar='S',
                    help='Decide whether to remove obvious positive sample before training (default: No)')                      



# Light GBM
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'min_data_in_leaf': 5,
    'num_leaves': 128,
    'learning_rate': 0.015,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity':-2,
    'silent': 1
}
# XGBoost parameters 
xgb_params = {
    'booster':'gbtree',
    'objective':'binary:logistic',
    'stratified':True,
    'scale_pos_weights':19,
    'max_depth':7,
    'min_child_weight':3,
    'gamma':1,
    'subsample':0.5,
    'colsample_bytree':0.5,
    'lambda':1,
    'eta':0.02,
    'seed':20,
    'silent':1
}



if __name__ == '__main__':
    
    args = parser.parse_args()
    mode = 'P'
    if args.Remove == 'Yes':
        mode = 'R'
    train_x, train_y, test_x, testIndex, trainFs, trainLs, trainIds = readFeature(Mode=mode)
    
    SEED = 1024
    feature_fold = args.FeatureFold
    nfeature = train_x.shape[1]
    for j in range(args.FeatureLoop):
        seed = SEED + np.random.randint(0, 200) * j
        kf = KFold(n=nfeature, n_folds=feature_fold, shuffle=True, random_state=seed)
        for k, (feeMoreIndex, feeLessIndex) in enumerate(kf):
            # feature split
            if k != 0:
                continue
            if args.Part == 'More':
                feeIndex = feeMoreIndex
            else:
                feeIndex = feeLessIndex
            Subtr_X, Subte_X = train_x.iloc[:, feeIndex], test_x.iloc[:, feeIndex]
            print('### Gnerating Result on %d Feature %d Sample###' % (Subtr_X.shape[1], Subtr_X.shape[0]))
            
            SubtrainFs = []
            for trainF in trainFs:
                SubtrainFs.append(trainF[:, feeIndex])
            frame1 = MLframe(Subtr_X, train_y, Subte_X, SubtrainFs, trainLs, trainIds)
            # bagging model
            for i in range(args.BaggingLoop):
                curr_seed = SEED + i * i + j + np.random.randint(0, 200)
                xgb = XgbWrapper(seed=curr_seed, params=xgb_params, rounds =601, has_weight = False)
                lgb = LgbWrapper(seed=curr_seed, params=lgb_params, rounds = 601, has_weight = False)
                
                cvfold = args.CVFold
                if args.ARCH == 'xgb':
                    Fscore, predict = frame1.cv(xgb, cvfold, how='mine')
                    name = 'xgb'

                if args.ARCH == 'lgb':
                    Fscore, predict = frame1.cv(lgb, cvfold, how='mine')
                    name = 'lgb'

                curr = time.strftime('%d-%H-%M', time.localtime())
                formatStr = '%s-%s-%d-%.5f'
                if args.Remove == 'Yes':
                    formatStr = 'R-' + formatStr
                store(predict, testIndex, formatStr % (curr, name, curr_seed, Fscore))
    

    