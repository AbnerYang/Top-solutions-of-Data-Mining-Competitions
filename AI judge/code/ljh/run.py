#-*- encoding:utf-8 -*-
from penalty import *
from laws import *
import processing 
import feature


def process():
	##数据预处理
	processing.run()

	# ##特征提取
	print 'getting features..'
	feature.run()

	##罚金的模型
	getCNNpenalty()
	getDenseCNNpenalty()

	##法文的模型
	getCNNlaws()
	getWideCNNlaws()

if __name__ == '__main__':
	process()
	
