# -*- encoding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
import jieba
import warnings
import codecs
import copy
from tqdm import tqdm
from collections import defaultdict
import random as rn
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Merge,concatenate,Reshape
from keras.layers import Convolution1D,Convolution2D, MaxPooling1D, Embedding,BatchNormalization,Dropout,MaxPooling2D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling1D,GlobalMaxPooling2D,GlobalAveragePooling1D
from keras.callbacks import EarlyStopping,LearningRateScheduler,ModelCheckpoint
from keras import regularizers
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import Adamax, RMSprop, Adam
from keras.layers import Activation,add
from sklearn.metrics import f1_score
import copy
import re
import h5py
import gc
import time
warnings.filterwarnings('ignore')

def getLawLabel(df):
    data = df['laws'].values
    n = df.shape[0]
    matrix = np.zeros((n,452))
    for i,laws in enumerate(data):
        seq = laws.split(',')
        for l in seq:
            try:
                matrix[i,int(l)-1] = 1
            except IndexError:
                print laws
    return matrix

def evalF1(p,l):
    preds = copy.deepcopy(p)
    label = copy.deepcopy(np.argmax(l, axis = 1))
    score = f1_score(label, preds, average='weighted')
    return score

def Jaccard_Sim(y_true,y_pred):
    y_pred = K.greater_equal(y_pred,0.5)
    y_pred = K.cast(y_pred,dtype='float32')
    intersection = K.sum(y_true*y_pred,axis=1)
    pred = K.sum(y_pred,axis=1)
    true = K.sum(y_true,axis=1)
    union = pred + true - intersection
    jaccard = intersection / (union+ K.epsilon())
    jaccard = K.mean(jaccard)
    return jaccard


def getData(config):
	log('Read data..')
	train = pd.read_csv('../../feature/yyt/traindoc_money_law.csv')
	test = pd.read_csv('../../feature/yyt/testdoc_money_law.csv')
	labels = getLawLabel(train)

	log('Tokenizer..')
	#'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	MAX_NB_WORDS = config['MAX_NB_WORDS']
	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts(train['doc'].append(test.doc))
	vocab = tokenizer.word_index
	num_words = min(MAX_NB_WORDS, len(vocab))
	
	log('Split Data..')
	VALIDATION_SPLIT = config['VALIDATION_SPLIT']
	split_seed = config['SPLIT_SEED']
	X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size=VALIDATION_SPLIT, random_state=split_seed)


	log('Get word2id..')
	X_train_word_ids = tokenizer.texts_to_sequences(X_train.doc)
	X_val_word_ids = tokenizer.texts_to_sequences(X_val.doc)

	X_online_train_word_ids = tokenizer.texts_to_sequences(train.doc)
	X_test_word_ids = tokenizer.texts_to_sequences(test.doc)

	log('Get Padding..')
	INPUT_LEN = config['INPUT_LEN']
	x_train = pad_sequences(X_train_word_ids, maxlen=INPUT_LEN,padding='post')
	x_val = pad_sequences(X_val_word_ids, maxlen=INPUT_LEN,padding='post')

	x_online_train = pad_sequences(X_online_train_word_ids, maxlen=INPUT_LEN,padding='post')
	x_test = pad_sequences(X_test_word_ids, maxlen=INPUT_LEN,padding='post')
	log('Return LabelEncoder..')

	ID = test[['ID']].reset_index(drop = True)
	return x_train, x_val, y_train, y_val, x_online_train, x_test, labels,ID



def TextCNN(config, model_view = True):
    inputs = Input(shape=(config['INPUT_LEN'],), dtype='int32',name = 'input')
    embedding_cnn = Embedding(config['MAX_NB_WORDS']+1,config['EMBEDDING_DIM'], input_length=config['INPUT_LEN'])(inputs)
    conv_out = []
    for filter in [1,2,3,4,5,6]:
        conv1 = Convolution1D(256,filter, padding='same')(embedding_cnn)
        conv1 = Activation('relu')(conv1)
        x_conv1 = GlobalMaxPooling1D()(conv1)
        conv_out.append(x_conv1)

    x = concatenate(conv_out)
    x = Dense(128)(x) 
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(452)(x)
    outputs = Activation('sigmoid',name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy',optimizer= 'adamax' ,metrics=[Jaccard_Sim])
    if model_view:
        model.summary()
    return model

def get_lr(epoch):
    if epoch < 4:
        return 0.0005
    elif epoch < 8:
        return 0.0002
    elif epoch < 12:
        return 0.0001
    else:
        return 0.00008

def log(stri):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print str(now)+' '+str(stri)


def run():
	config = {}
	config['MAX_NB_WORDS'] = 100000
	config['EMBEDDING_DIM'] = 256
	config['INPUT_LEN'] = 1100
	config['SPLIT_SEED'] = 30
	config['VALIDATION_SPLIT'] = 0.1
	log(config)
	log('Get DataSet..')
	local_train_x, local_test_x, local_train_y, local_test_y, online_train_x, online_test_x, online_train_y,ID = getData(config)
	gc.collect()
	lr = LearningRateScheduler(get_lr)
	np.random.seed(30)
	#--get Local Test
	log('Local Validate Model...')
	# model = TextCNN(config, model_view = True)
	# model.fit(local_train_x,local_train_y, batch_size=100, epochs=30,
	# 		callbacks = [EarlyStopping(monitor='Jaccard_Sim',patience=2,mode = 'max'), lr],
	# 		validation_data=(local_test_x, local_test_y))
	# del model
	#--get Online Result
	log('Online Train Model With Different Seed...')
	for i in range(10):
		log('Seed_%d'%i)
		lr = LearningRateScheduler(get_lr)
		np.random.seed(i)
		# model_check = ModelCheckpoint(filepath='../model/textCNN_money_all_seed.'+str(i)+'.hdf5',mode = 'min',monitor='val_loss', save_best_only=False, verbose=1)
		model = TextCNN(config, model_view = False)
		model.fit(online_train_x,online_train_y, 
		          batch_size=100, 
		          epochs=20,
		          callbacks = [lr])
		model.save('../../model/yyt/laws/textCNN_laws_all_seed.'+str(i)+'.hdf5')
		predict_online = model.predict(online_test_x, batch_size=100)
		pd.DataFrame(predict_online).to_csv('../../result/yyt/laws/textCNN_laws_all_%d'%(i), index = False, header = False, float_format = '%.8f')
		if i == 0:
			res = predict_online
		else:
			res += predict_online
		del model
		gc.collect()
	res = pd.DataFrame(res/10)
	res = pd.concat([ID, res], axis = 1)
	res.to_csv('../../result/yyt/laws/textCNN_laws_all_prob_blend.csv', index = False, header = False, float_format = '%.8f')
		


