# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
import warnings
import codecs
import copy
from tqdm import tqdm
import textCNN_money
import textCNN_laws
import preprocess

if __name__ == '__main__':
	textCNN_laws.log('preprocess...')
	preprocess.run()
	textCNN_laws.log('get laws result...')
	textCNN_laws.run()
	textCNN_laws.log('get money result...')
	textCNN_money.run()
