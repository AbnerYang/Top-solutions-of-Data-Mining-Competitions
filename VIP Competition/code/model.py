# -*- encoding:utf-8 -*-
import numpy as np 
import pandas as pd 
import copy
from frame import *
import xgboost as xgb
import lightgbm as lgb
import random

def evalRmse(preds,dtrain):
	N = 5
	label = dtrain.get_label()
	preds = np.around(preds,3)
	pos_mask = label==1
	pos_label = label[pos_mask]
	pos_preds = preds[pos_mask]
	n = len(pos_label)
	neg_label = label[~pos_mask]
	neg_preds = preds[~pos_mask]
	pos_res = (pos_label-pos_preds)**2
	neg_res = (neg_label-neg_preds)**2
	result = []
	for i in range(N):
		neg_spl = random.sample(neg_res,n)
		res = sum(pos_res+neg_spl)/n/2
		result.append(res**0.5)
	return 'RMSE',np.mean(result)


class XgbWrapper(object):
	def __init__(self, seed=0, params=None, rounds = 100, has_weight = False):
		self.param = params
		self.param['seed'] = seed
		self.nrounds = rounds
		self.has_weight = has_weight

	def train(self, x_train, y_train):
		if self.has_weight == True:
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
		if self.has_weight == True:
			print 'has weight...'
			weight = get_weight(y_train)
			dtrain = xgb.DMatrix(x_train, label=y_train, weight = weight)
		else:
			dtrain = xgb.DMatrix(x_train, label=y_train)

		xgb.cv(self.param, dtrain, self.nrounds, nfold = nfold, verbose_eval = 5, feval = evalRmse)
	
class LgbWrapper(object):
	def __init__(self, seed=0, params=None, rounds = 100, has_weight = False):
		self.param = params
		self.param['seed'] = seed
		self.nrounds = rounds
		self.has_weight = has_weight

	def train(self, x_train, y_train):
		if self.has_weight == True:
			print 'has weight...'
			weight = get_weight(y_train)
			dtrain = lgb.Dataset(x_train, label=y_train, weight = weight)
		else:
			dtrain = lgb.Dataset(x_train, label=y_train)
		self.gbdt = lgb.train(self.param, dtrain, num_boost_round = self.nrounds, valid_sets = [dtrain], verbose_eval = 50 )


	def train_test(self, x_train, y_train, x_test, y_test):
		if self.has_weight == True:
			print 'has weight...'
			weight =  get_weight(y_train)
			dtrain = lgb.Dataset(x_train, label=y_train, weight = weight)
		else:
			dtrain = lgb.Dataset(x_train, label=y_train)
		dtest = lgb.Dataset(x_test, label = y_test)
		self.gbdt = lgb.train(self.param, dtrain, self.nrounds, valid_sets = [dtest] , verbose_eval = 1)


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

		lgb.cv(self.param, dtrain, self.nrounds, nfold = nfold, feval=evalmape_lgb, verbose_eval = 50)
