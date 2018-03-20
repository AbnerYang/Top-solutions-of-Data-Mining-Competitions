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
import keras
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

from keras import backend as K


MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 100000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
INPUT_LEN = 1000
PROB_PATH = '../../result/ljh/money/'
FEATURE_PATH = '../../feature/ljh/'
  
config = {
  'status':'online',
  'mode':'penalty',
  'word_embed':False,
  'check_jaccard':False,
  }


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


def textCNNmodel(config): 

  embedding_layer = Embedding(MAX_NB_WORDS+1,EMBEDDING_DIM, 
                    embeddings_initializer=initializers.he_uniform(20),
                        input_length=INPUT_LEN,
                        trainable=True)

  model1 = Sequential()
  model1.add(embedding_layer)
  model1.add(Convolution1D(128, 2,padding='same'))
  model1.add(GlobalMaxPooling1D())
 
  model2= Sequential()
  model2.add(embedding_layer)
  model2.add(Convolution1D(128, 3,padding='same'))
  model2.add(GlobalMaxPooling1D())
 
  model3 = Sequential()
  model3.add(embedding_layer)
  model3.add(Convolution1D(128, 5,padding='same'))
  model3.add(GlobalMaxPooling1D())
  
  model4 = Sequential()
  model4.add(embedding_layer)
  model4.add(Convolution1D(128, 7,padding='same'))
  model4.add(GlobalMaxPooling1D())

  model = Sequential()
  model.add(Merge([model1,model2,model3,model4],mode='concat',concat_axis=1))
  model.add(Dropout(0.3))
  model.add(Dense(128,activation='relu'))

  if config['mode']=='penalty':
    	model.add(Dense(8,activation='softmax'))
    	model.compile(loss='categorical_crossentropy',
                  	optimizer='adamax',
                  	metrics=[macro_f1])
  else:
    model.add(Dense(452,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
              	optimizer='adam',
              	metrics=[Jaccard_Sim])
  print model.summary()
  return model


def k_maxpooling(conv, topk, dim):
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=topk)
        return tf.reshape(k_max[0], (-1, dim * topk))
    k_max = Lambda(_top_k, output_shape=(dim * topk,))(conv)
    return k_max
  

def add_layer(L, outdim=32):
    c = BatchNormalization()(L)
    c = Dense(outdim)(c)
    c = PReLU()(c)
    L = Merge(mode='concat', concat_axis=-1)([L, c])
    return L


def dense_cnn_model():
    my_embedding = Embedding(input_dim=MAX_NB_WORDS+1, output_dim=EMBEDDING_DIM, input_length=None) #128 
    #---------keyword 1 -------------------------
    in1 = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    emb1 = my_embedding(in1)
    
    cnn1 = Convolution1D(filters=256, kernel_size=7, kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1) # relu   
  
    x1 = GlobalMaxPooling1D()(cnn1)

    
    cnn3 = Convolution1D(filters=256, kernel_size=3,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
    x3 = GlobalMaxPooling1D()(cnn3)
    
#        
    cnn5 = Convolution1D(filters=256, kernel_size=5,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
    x5 = GlobalMaxPooling1D()(cnn5)
    # cnn4 = Convolution1D(filters=256, kernel_size=2,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
    x1 = Merge(mode='concat', concat_axis=-1)([x1, x3, x5])

    #block1
    for i in range(4):
        x1 = add_layer(x1, 128) #128
    x1 = BatchNormalization()(x1)
    x1 = PReLU()(x1)
    x1 = Dense(128)(x1)
 
    #block2
    for i in range(4):
        x1 = add_layer(x1, 128)
    #x1 = BatchNormalization()(x1)
    #x1 = Dense(128)(x1)

    x = BatchNormalization()(x1)
    x = Dense(256)(x) #128
    
  
    x = PReLU()(x)
    x = Dropout(0.35)(x)  #0.25
    y = Dense(8, activation='softmax')(x)
    #y = Dense(8, activation='sigmoid')(x)

    #model = Model(inputs=[in1, in2], outputs=y)
    model = Model(inputs=[in1], outputs=y)
    rmsprop = keras.optimizers.Adadelta(lr=1.0, rho=0.9, epsilon=1e-06) #lr=1.0 rho=0.95
    model.compile(optimizer=rmsprop,  loss='categorical_crossentropy',   metrics=[macro_f1])
    print model.summary()
    return model


def getData():
  train = pd.read_csv(FEATURE_PATH+'train.tsv',header=None)
  test = pd.read_csv(FEATURE_PATH+'test.tsv',header=None)
  testIndex = test[[0]]
  testIndex.columns = ['ID']

  x_train,x_test,y_train = train[range(3,1003)].values,test[range(3,1003)].values,train[1]
  y_labels = list(y_train.value_counts().index)
  le = LabelEncoder()
  le.fit(y_labels)
  num_labels = len(y_labels)
  y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)

  return x_train,x_test,y_train,testIndex

def getCNNpenalty():
  X_train,X_test,Y_train,testIndex = getData()
  print X_train.shape,X_test.shape

  TIMES = 8 #10
  for SEED in range(TIMES):
    print SEED
    M = SEED
    #X_train, X_val, Y_train, Y_val = train_test_split(train[range(2,1002)], train[1], test_size=VALIDATION_SPLIT, random_state=SEED)
    #X_train, X_val, Y_train, Y_val = train_test_split(pd.DataFrame(x_train), y_train, test_size=VALIDATION_SPLIT, random_state=SEED)
    np.random.seed(SEED)
    #model = getModel()
    #model = TextCNN(num_words, EMBEDDING_DIM, INPUT_LEN,config)
    model = textCNNmodel(config)
    #model_check = ModelCheckpoint(filepath='models/seed.'+str(SEED)+'.{val_Jaccard_Sim:.4f}.weights.{epoch:02d}.hdf5', save_best_only=True, verbose=1) 
   #model.fit([X_train],laws_train, 
   ##best epochs is 4
    print 'training cnn model..'
    model.fit([X_train],Y_train, 
              batch_size=16, 
              epochs=4,)
              #callbacks = [change_lr],
              #callbacks=snapshot.get_callbacks(model_prefix=model_prefix),
              #callbacks = [EarlyStopping(monitor='val_Jaccard_Sim',patience=0,mode='max'),change_lr],
              #validation_data=([X_val.values,wide_val], Y_val)) 
    #mode = 'model NOT updated!!!'
    # print 'saving model',SEED
    # model.save('models/1203_penalty_cnn_'+str(SEED)+'.h5')
    print 'predicting test..'
    predict = model.predict(X_test)
    if SEED==0:
      pred = predict
    else:
      pred +=predict
    del model 


  # print 'storing result..'
  # res = np.argmax(pred,axis=1)
  # storeResult(testID, res, '1203-penalty-cnn*10')
  df = pd.DataFrame(pred)
  sums = df.sum(axis=1)
  for col in df.columns:
    df[col] /=sums
  df = testIndex.join(df)
  df.to_csv(PROB_PATH+'jh_penalty_cnn_blending_prob.csv',index=0,header=None,float_format = '%.6f')
  print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

def getDenseCNNpenalty():
  X_train,X_test,Y_train,testIndex = getData()
  print X_train.shape,X_test.shape
  TIMES = 8#10
  for SEED in range(TIMES):
    print SEED

    np.random.seed(SEED)
    #model = getModel()
    #change_lr = LearningRateScheduler(scheduler)  
    #model = TextCNN(num_words, EMBEDDING_DIM, INPUT_LEN,config)
    model = dense_cnn_model()    #model_check = ModelCheckpoint(filepath='models/seed.'+str(SEED)+'.{val_Jaccard_Sim:.4f}.weights.{epoch:02d}.hdf5', save_best_only=True, verbose=1) 
   #model.fit([X_train],laws_train, 
   ##best epochs is 4
    print 'training dense cnn model..'
    model.fit([X_train],Y_train, 
              batch_size=16, 
              epochs=11,)
              #callbacks = [change_lr],
              #callbacks=snapshot.get_callbacks(model_prefix=model_prefix),
              #callbacks = [EarlyStopping(monitor='val_Jaccard_Sim',patience=0,mode='max'),change_lr],
              #validation_data=([X_val.values], Y_val)) 
    #mode = 'model NOT updated!!!'
    # print 'saving model',SEED
    # model.save('models/1210_penalty_dense_'+str(SEED)+'.h5')
    print 'predicting test..'
    predict = model.predict(X_test)
    if SEED==0:
      pred = predict
    else:
      pred +=predict
    del model 
  # print 'storing result..'
  # res = np.argmax(pred,axis=1)
  # storeResult(testID, res, '1210-penalty-dense*5')

  df = pd.DataFrame(pred)
  # df.to_csv('result/1210-densenet-penalty-5-blending_prob.csv',index=0,header=None)
  sums = df.sum(axis=1)
  for col in df.columns:
    df[col] /=sums
  df = testIndex.join(df)
  df.to_csv(PROB_PATH +'jh_penalty_dense_cnn_blending_prob.csv',index=0,header=None,float_format = '%.6f')

