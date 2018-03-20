# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
import jieba
import warnings
import codecs
from collections import defaultdict
#import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Merge,concatenate,Activation,Lambda
from keras.layers import Convolution1D, MaxPooling1D, Embedding,BatchNormalization,Dropout
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.models import Model
from keras.layers.pooling import GlobalMaxPooling1D
from keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers.advanced_activations import PReLU
from keras import optimizers
from keras import initializers

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
LAW_COUNT = 452
INPUT_LEN = 1000
JACCARD_TH = 0.5
PROB_PATH = '../../result/ljh/laws/'
FEATURE_PATH = '../../feature/ljh/'





def getLawLabel(data):
    #data = df[2].values
    #n = df.shape[0]
    n = len(data)
    matrix = np.zeros((n,LAW_COUNT))
    for i,laws in enumerate(data):
        seq = laws.split(',')
        for l in seq:
            try:
                matrix[i,int(l)-1] = 1
            except IndexError:
                print laws
    return matrix



from keras import backend as K
def macro_f1(y_true, y_pred):
    true_positives = K.sum(y_true * K.one_hot(K.argmax(y_pred),8), axis = 0)
    predicted_positives = K.sum(K.one_hot(K.argmax(y_pred),8), axis = 0)
    precision = true_positives / (predicted_positives + K.epsilon())
    
    true_positives = K.sum(y_true * K.one_hot(K.argmax(y_pred),8), axis = 0)
    possible_positives = K.sum(y_true, axis = 0)
    recall = true_positives / (possible_positives + K.epsilon())
    
    f1 = 2*((precision*recall)/(precision+recall+ K.epsilon()))
    macro_f1 = K.sum((K.sum(y_true, axis = 0)*f1)/(K.sum(y_true)+ K.epsilon()))
    return macro_f1

def Jaccard_Sim(y_true,y_pred):
    y_pred = K.greater_equal(y_pred,JACCARD_TH)
    y_pred = K.cast(y_pred,dtype='float32')
    intersection = K.sum(y_true*y_pred,axis=1)
    pred = K.sum(y_pred,axis=1)
    true = K.sum(y_true,axis=1)
    union = pred + true - intersection
    jaccard = intersection / (union+ K.epsilon())
    jaccard = K.mean(jaccard)
    return jaccard

    #prelu = PReLU()
def cnnModel():
  embedding_layer = Embedding(MAX_NB_WORDS+1,EMBEDDING_DIM, 
                embeddings_initializer=initializers.he_uniform(20),
                    #weights=[embedding_matrix],
                    input_length=INPUT_LEN,
                    trainable=True)
#                     #trainable=False))

  model1 = Sequential()
  model1.add(embedding_layer)
  model1.add(Convolution1D(128, 4,padding='same',init='he_normal'))
  model1.add(BatchNormalization()) 
  model1.add(Activation('relu'))
  model1.add(Convolution1D(128, 4, padding='same',activation='relu',init='he_normal'))
  model1.add(GlobalMaxPooling1D())

  model2= Sequential()
  model2.add(embedding_layer)
  model2.add(Convolution1D(128, 3,padding='same',init='he_normal'))
  model2.add(BatchNormalization()) 
  model2.add(Activation('relu'))
  model2.add(Convolution1D(128, 3, padding='same',activation='relu',init='he_normal'))
  model2.add(GlobalMaxPooling1D())

  model3 = Sequential()
  model3.add(embedding_layer)
  model3.add(Convolution1D(128, 5,padding='same',init='he_normal'))
  model3.add(BatchNormalization()) 
  model3.add(Activation('relu'))
  model3.add(Convolution1D(128, 5, padding='same',activation='relu',init='he_normal'))
  model3.add(GlobalMaxPooling1D())

  model4 = Sequential()
  model4.add(embedding_layer)
  model4.add(Convolution1D(128, 7,padding='same',init='he_normal'))
  model4.add(BatchNormalization()) 
  model4.add(Activation('relu'))
  model4.add(Convolution1D(128, 7, padding='same',activation='relu',init='he_normal'))
  model4.add(GlobalMaxPooling1D())


  model = Sequential()
  model.add(Merge([model1,model2,model3,model4],mode='concat',concat_axis=1))
#model.add(GRU(128, dropout=0.2, recurrent_dropout=0.1))
  model.add(Dropout(0.3))
  model.add(Dense(128,activation='relu',init='he_normal'))
  model.add(Dense(LAW_COUNT,activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
              optimizer='adamax',
               metrics=[Jaccard_Sim])
  model.summary()
  return model

def widecnnModel():
  embedding_layer = Embedding(MAX_NB_WORDS+1,EMBEDDING_DIM, 
                embeddings_initializer=initializers.he_uniform(20),
                    #weights=[embedding_matrix],
                    input_length=INPUT_LEN,
                    trainable=True)
#                     #trainable=False))

  model1 = Sequential()
  model1.add(embedding_layer)
  model1.add(Convolution1D(128, 4,padding='same',init='he_normal'))
  model1.add(BatchNormalization()) 
  model1.add(Activation('relu'))
  model1.add(Convolution1D(128, 4, padding='same',activation='relu',init='he_normal'))
  model1.add(GlobalMaxPooling1D())

  model2= Sequential()
  model2.add(embedding_layer)
  model2.add(Convolution1D(128, 3,padding='same',init='he_normal'))
  model2.add(BatchNormalization()) 
  model2.add(Activation('relu'))
  model2.add(Convolution1D(128, 3, padding='same',activation='relu',init='he_normal'))
  model2.add(GlobalMaxPooling1D())

  model3 = Sequential()
  model3.add(embedding_layer)
  model3.add(Convolution1D(128, 5,padding='same',init='he_normal'))
  model3.add(BatchNormalization()) 
  model3.add(Activation('relu'))
  model3.add(Convolution1D(128, 5, padding='same',activation='relu',init='he_normal'))
  model3.add(GlobalMaxPooling1D())

  model4 = Sequential()
  model4.add(embedding_layer)
  model4.add(Convolution1D(128, 7,padding='same',init='he_normal'))
  model4.add(BatchNormalization()) 
  model4.add(Activation('relu'))
  model4.add(Convolution1D(128, 7, padding='same',activation='relu',init='he_normal'))
  model4.add(GlobalMaxPooling1D())
  
  #prelu = PReLU()
  wide = Sequential()
  wide.add(Dense(512,input_shape=(WIDE_LEN,),activation='tanh',init='he_normal' ))
  wide.add(Dropout(0.3))
  wide.add(BatchNormalization()) 
  wide.add(Dense(32,activation='tanh',init='he_normal')) #128
  #wide.add(Dropout(0.3))
  wide.add(BatchNormalization()) 
  wide.add(Dense(8,activation='tanh',init='he_normal'))


  model = Sequential()
  model.add(Merge([model1,model2,model3,model4,wide],mode='concat',concat_axis=1))
  model.add(Dropout(0.3))
  model.add(Dense(128,activation='relu',init='he_normal'))
  model.add(Dense(LAW_COUNT,activation='sigmoid'))
  model.compile(loss='binary_crossentropy',
              optimizer='adamax',
               metrics=[Jaccard_Sim])
  model.summary()
  return model


def getData():
  train = pd.read_csv(FEATURE_PATH+'train.tsv',header=None)
  test = pd.read_csv(FEATURE_PATH+'test.tsv',header=None)
  wide_train = pd.read_csv(FEATURE_PATH+'train_feature.csv')
  wide_test = pd.read_csv(FEATURE_PATH+'test_feature.csv')
  print wide_train.shape
  wide_train.drop(['id','penalty'],axis=1,inplace=True)
  wide_test.drop(['id',],axis=1,inplace=True)
  x_train,x_test,y_train = train[range(3,1003)].values,test[range(3,1003)].values,train[2]
  y_train = getLawLabel(y_train)
  testIndex=  test[[0]]
  testIndex.columns = ['ID']
  return x_train, x_test, y_train, wide_train.values, wide_test.values, testIndex



def getCNNlaws():
  def scheduler(epoch):  
    epo = [2,5,7,12,20]
    #epo = [2,5,15,30,40]
    lrs = [.001,.0002,.00002,.000002,.0000001]
    for i,e in enumerate(epo):
      if epoch==epo:
          K.set_value(model.optimizer.lr, lrs[i]) 
      return K.get_value(model.optimizer.lr)
  X_train, X_test, Y_train, wide_train, wide_test,testIndex = getData()
  WIDE_LEN = wide_train.shape[1]
  TIMES = 8 #10
  for SEED in range(TIMES):
    print SEED
    np.random.seed(SEED)
    #model = getModel()
    
    change_lr = LearningRateScheduler(scheduler)  
    #model = TextCNN(num_words, EMBEDDING_DIM, INPUT_LEN,config)
    model = cnnModel()
    print 'training cnn model..'
    #model_check = ModelCheckpoint(filepath='models/seed.'+str(SEED)+'.{val_Jaccard_Sim:.4f}.weights.{epoch:02d}.hdf5', save_best_only=True, verbose=1) 
    model.fit([X_train],Y_train, 
    #model.fit([train[range(2,1002)]],law_label, 
              batch_size=16, 
              epochs=6,
              callbacks = [change_lr])
              #callbacks=snapshot.get_callbacks(model_prefix=model_prefix),
              #callbacks = [EarlyStopping(monitor='val_Jaccard_Sim',patience=5,mode='max'),change_lr],
              #validation_data=([X_val,wide_val], laws_val)) 
    #mode = 'model NOT updated!!!'
  #   print 'saving model',SEED
  #   model.save('models/1202_cnn_'+str(SEED)+'.h5')
    print 'predicting test..'
    predict = model.predict(X_test)
    if SEED==0:
      pred = predict
    else:
      pred +=predict
    #del model
  print 'storing cnn prob..'
  df = pd.DataFrame(pred)
  for col in df.columns:
    df[col] /= TIMES 
  df = testIndex.join(df)
  df.to_csv(PROB_PATH+'jh_laws_cnn_blending_prob.csv',index=0,header=None,float_format = '%.6f')

  print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

def getWideCNNlaws():
  X_train, X_test, Y_train, wide_train, wide_test,testIndex = getData()
  global WIDE_LEN
  WIDE_LEN = wide_train.shape[1]
  def scheduler(epoch):  
  #epo = [2,5,7,12,20]
    epo= [2,5,15,30,40]
    lrs = [.001,.0002,.00002,.000002,.0000001]
    for i,e in enumerate(epo):
        if epoch==epo:
            K.set_value(model.optimizer.lr, lrs[i]) 
        return K.get_value(model.optimizer.lr)

  TIMES = 8#10
  for SEED in range(TIMES):
    print SEED
    np.random.seed(SEED)
    #model = getModel()
    change_lr = LearningRateScheduler(scheduler)  
    #model = TextCNN(num_words, EMBEDDING_DIM, INPUT_LEN,config)
    model = widecnnModel()
    #model_check = ModelCheckpoint(filepath='models/seed.'+str(SEED)+'.{val_Jaccard_Sim:.4f}.weights.{epoch:02d}.hdf5', save_best_only=True, verbose=1) 
    #model.fit([X_train,wide_train],laws_train, 
    model.fit([X_train,wide_train],Y_train, 
              batch_size=16, 
              epochs=8,
              callbacks = [change_lr])
              #callbacks=snapshot.get_callbacks(model_prefix=model_prefix),
              #callbacks = [EarlyStopping(monitor='val_Jaccard_Sim',patience=5,mode='max'),change_lr],
              #validation_data=([X_val,wide_val], laws_val)) 
    #mode = 'model NOT updated!!!'
    #print 'saving model',SEED
    #model.save('models/1209_wide&cnn_'+str(SEED)+'.h5')
    print 'predicting test..'
    predict = model.predict([X_test,wide_test])
    if SEED==0:
      pred = predict
    else:
      pred +=predict
    #del model
    
  df = pd.DataFrame(pred)
  for col in df.columns:
    df[col] /= TIMES 
  df = testIndex.join(df)
  df.to_csv(PROB_PATH+'jh_laws_wide&cnn_blending_prob.csv',index=0,header=None,float_format = '%.6f')