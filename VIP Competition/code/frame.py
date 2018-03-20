# -*- encoding:utf-8 -*-
import numpy as np 
import pandas as pd 
import time
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold,KFold

def LogInfo(stri):
	print str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+' '+stri

class MLframe(object):
	"""docstring for MLframe"""
	def __init__(self, train_x, train_y, test_x, seed = 0):
		self.seed = seed
		self.train_x = train_x
		self.train_y = train_y
		self.test_x = test_x
		
	def cv(self, model, nfold = 5,isxgb = True):
		if isxgb == True:
			model.default_cv(self.train_x, self.train_y, nfold)

	def stacking(self, clf, nfold):
		oof_train = np.zeros([self.train_x.shape[0]])
		oof_test = np.zeros([self.test_x.shape[0]])
		oof_test_skf = np.empty([self.test_x.shape[0], nfold])
		skf = list(StratifiedKFold(n_splits = nfold, random_state = self.seed).split(self.train_x, self.train_y))
		for i, (train_index, test_index) in enumerate(skf):
			LogInfo("--fold:"+str(i))
			x_tr = self.train_x.iloc[train_index,:]
			y_tr = self.train_y[train_index]
			x_te = self.train_x.iloc[test_index,:]
			print x_tr.shape, x_te.shape

			clf.train(x_tr, y_tr)

			oof_train[test_index] = clf.predict(x_te)
			oof_test_skf[:, i] = clf.predict(self.test_x)

		evalu = evaluate(oof_train, self.train_y)
		print evalu.rmse()
		oof_test = oof_test_skf.mean(axis=1).reshape(self.test_x.shape[0])
		print oof_test
		return oof_train, oof_test

	def train_test(self, clf, test_y, isxgb = False):
		if isxgb == True:
			clf.train_test(self.train_x, self.train_y, self.test_x, test_y)
		else:
			clf.train(self.train_x, self.train_y)
			predict = clf.predict(self.test_x)
			evalu = evaluate(predict, test_y)
			print evalu.rmse()
		return clf

	def train_predict(self, clf):
		clf.train(self.train_x, self.train_y)
		predict = clf.predict(self.test_x)
		return clf, predict

	def get_importance(self, clf, name):
		# print featureName[0:]
		# clf.train(self.train_x, self.train_y)
		imp_f = clf.feature_importances()
		df = pd.DataFrame({'feature':[key for key, value in imp_f.items()], 'fscore':[value for key, value in imp_f.items()]})
		# mean_imp = np.mean(df['fscore'])
		# top_f = df.loc[df['fscore'] >= mean_imp]
		df['fscore'] = df['fscore'] / df['fscore'].sum()
		df = df.sort_values(by = 'fscore',ascending=False)
		df.to_csv('../imp/'+name+'.csv', index = False)



class evaluate(object):
	"""docstring for evaluate"""
	def __init__(self, predict, true):
		self.predict = predict
		self.true = true

	def rmse(self, preds, label):
		N = 5
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
		return np.mean(result)