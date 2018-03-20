# -- encoding:utf-8 --
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import copy
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

def view_mutil_result(data):
	plt.figure()
	t = range(data.shape[0])
	colName = data.columns
	for i in range(data.shape[1]):
		plt.plot(t,data.values[:,i],"-",label=colName[i])

	plt.xlabel("t")
	plt.ylabel("values")
	plt.title("view_mutil_result")

	plt.grid(True)
	plt.legend()
	plt.show()

def view_semi_result(data):
	print data.astype(float).corr()
	colormap = plt.cm.viridis
	plt.figure(figsize = (data.shape[1], data.shape[1]))
	plt.title('PCC-of Result', y = 1.05, size = 15)
	sns.heatmap(data.astype(float).corr(), linewidth = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True)
	plt.show()

def result_blend():
	r1 = pd.read_csv('../../result/xgb-1.csv', header = 0)
	r2 = pd.read_csv('../../result/xgb-2.csv', header = 0)
	r3 = pd.read_csv('../../result/nn-1.csv', header = 0)
	r4 = pd.read_csv('../../result/xgb-bagging.csv', header = 0)
	
	r1.columns = ['intersection_id','tollgate_id','time_window','xgb-1']
	r2.columns = ['intersection_id','tollgate_id','time_window','xgb-2']
	r3.columns = ['intersection_id','tollgate_id','time_window','nn-1']
	r4.columns = ['intersection_id','tollgate_id','time_window','bagging']

	r = pd.merge(r1, r2, on = ['intersection_id', 'tollgate_id', 'time_window'], how = 'left')
	r = pd.merge(r, r3, on = ['intersection_id', 'tollgate_id', 'time_window'], how = 'left')
	r = pd.merge(r, r4, on = ['intersection_id', 'tollgate_id', 'time_window'], how = 'left')
	
	r['avg_travel_time'] = r['nn-1'].values*0.45 + r['xgb-1'].values*0.35 + r['xgb-2'].values*0.15 + r['bagging'].values*0.05

	r.sort_values(by = ['intersection_id','tollgate_id','time_window']).to_csv('../../result/final-duibi-online.csv', index = False)
	
	r[['intersection_id','tollgate_id','time_window','avg_travel_time']].to_csv('../../result/final-blend-1.csv', index = False)
	r = r[['xgb-1','nn-1','xgb-2','bagging','avg_travel_time']]
	print r.T.mean(axis = 1)
	# view_semi_result(r)
	# view_mutil_result(r)


def compare():
	r1 = pd.read_csv('../../result/final-blend-1.csv', header = 0)
	r2 = pd.read_csv('../../result/final-blend.csv', header = 0)

	r1.columns = ['intersection_id','tollgate_id','time_window','new']
	r2.columns = ['intersection_id','tollgate_id','time_window','old']
	
	r = pd.merge(r1, r2, on = ['intersection_id', 'tollgate_id', 'time_window'], how = 'left')

	r = r[['new','old']]
	print r.T.mean(axis = 1)
	view_semi_result(r) 
	view_mutil_result(r)

if __name__ == '__main__':
	#-- get result
	result_blend()
	#--compare result with PCC and mean values
	compare()
