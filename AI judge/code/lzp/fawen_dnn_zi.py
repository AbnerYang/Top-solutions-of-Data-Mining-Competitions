# -*- coding: utf-8 -*-


import numpy as np

import keras
from keras.layers import Dense, Dropout, BatchNormalization, Convolution1D,\
GlobalMaxPooling1D, Embedding, Input, Merge, merge, Dot, dot, Reshape, Lambda, Masking, Activation, GlobalAveragePooling1D, PReLU, TimeDistributed
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K
from keras.layers.merge import Concatenate
from keras.layers.core import Flatten

import random
import pandas as pd
import numpy as np
import math
import gc
import tensorflow as tf
import os

#from Layers import ConvBlockLayer
#from utils import get_conv_shape

from keras.layers.pooling import MaxPooling1D
from sklearn.cross_validation import train_test_split


train_path = '../../feature/lzp/fawen.train.zi2.txt'
test_path = '../../feature/lzp/fawen.test.zi2.txt'

result_path = '../../result/lzp/laws/fawen.zi.tsv'
model_path = '../../model/lzp/laws/'


np.random.seed(666)

#1:6:42162
max_char_id = 5388 #9:32748 #4:54105 #2:81171   #84621

max_seq_len = 3000
epoch = 6
ts = 0
jac_thre = 0.5

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
    y_pred = K.greater_equal(y_pred, 0.5)
    y_pred = K.cast(y_pred,dtype='float32')
    intersection = K.sum(y_true*y_pred,axis=1)
    pred = K.sum(y_pred,axis=1)
    true = K.sum(y_true,axis=1)
    union = pred + true - intersection
    jaccard = intersection / (union+ K.epsilon())
    jaccard = K.mean(jaccard)
    return jaccard
    

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



def jaccard_distance_loss(y_true, y_pred, smooth=100):
#    y_pred = K.greater_equal(y_pred, 0.5)
#    y_pred = K.cast(y_pred,dtype='float32')
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jaccard = (intersection + smooth) / (sum_ - intersection + smooth)

#    y_pred = K.greater_equal(y_pred, 0.5)
#    y_pred = K.cast(y_pred,dtype='float32')
#    intersection = K.sum(y_true*y_pred,axis=1)
#    pred = K.sum(y_pred,axis=1)
#    true = K.sum(y_true,axis=1)
#    union = pred + true - intersection
#    jaccard = intersection / (union+ K.epsilon())
#    jaccard = K.mean(jaccard)
    
    return (1 - jaccard) * smooth*100

    
      
def build_model(train_x, train_y, test_x = []):
    #---共享
    my_embedding = Embedding(input_dim=max_char_id+3, output_dim=128, input_length=None) #128 
    
    
    #---------keyword 1 -------------------------
    in1 = Input(shape=(max_seq_len,), dtype='int32')
    emb1 = my_embedding(in1)
    
    
    sgsize = 4
    cnn1 = Convolution1D(filters=256, kernel_size=7, kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1) # relu
    t1 = TimeDistributed(Dense(sgsize, activation='sigmoid'))(cnn1)
    t1 = Flatten()(t1)
    t1 = Dense(256)(t1)
    t1 = Dropout(0.5)(t1)

    
#    tb = TimeDistributed(Dense(4, activation='sigmoid'))(cnn1)
#    tb = Flatten()(tb)
#    tb = Dense(256)(tb)
#    t1 = Merge(mode='mul', concat_axis=-1)([ta, tb])
    
    #xa = Flatten()(cnn1)
    
#    cnn1 = emb1
#    num_filters = [256, 256, 256]
#    #---------------
#    # Each ConvBlock with one MaxPooling Layer
#    for i in range(len(num_filters)):
#        cnn1 = ConvBlockLayer(get_conv_shape(cnn1), num_filters[i])(cnn1)
#        cnn1 = MaxPooling1D(pool_size=3, strides=2, padding="same")(cnn1)
    
    
    
    x1 = GlobalMaxPooling1D()(cnn1)
    #x1 = k_maxpooling(cnn1, 15, 256)
    
    #x1 = GlobalAveragePooling1D()(cnn1)
    

    cnn3 = Convolution1D(filters=256, kernel_size=3,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
    x3 = GlobalMaxPooling1D()(cnn3)
    t3 = TimeDistributed(Dense(sgsize, activation='sigmoid'))(cnn3)
    t3 = Flatten()(t3)
    t3 = Dense(256)(t3)
    t3 = Dropout(0.5)(t3)
    
#        
    cnn5 = Convolution1D(filters=256, kernel_size=5,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
    x5 = GlobalMaxPooling1D()(cnn5)
    t5 = TimeDistributed(Dense(sgsize, activation='sigmoid'))(cnn5)
    t5 = Flatten()(t5)
    t5 = Dense(256)(t5)
    t5 = Dropout(0.5)(t5)
    
#    
#    cnn4 = Convolution1D(filters=256, kernel_size=7,kernel_initializer = 'he_uniform', padding='valid', activation='relu')(emb1)
#    x4 = GlobalMaxPooling1D()(cnn4)    
#    x1 = Merge(mode='concat', concat_axis=-1)([x1, x2, x3, x4])
    
    x1 = Merge(mode='concat', concat_axis=-1)([x1, x3, x5])

    #block1
#    for i in range(4):
#        x1 = add_layer(x1, 128) #128
#    x1 = BatchNormalization()(x1)
#    x1 = PReLU()(x1)
#    x1 = Dense(128)(x1)
 
    #block2
    for i in range(4):
        x1 = add_layer(x1, 128)
    #x1 = BatchNormalization()(x1)
    #x1 = Dense(256)(x1)

       
    x = BatchNormalization()(x1)
    x = Dense(256)(x) #128
    

    #0.7987
    #t1s = Merge(mode='mul', concat_axis=-1)([x, t1])
    #t3s = Merge(mode='mul', concat_axis=-1)([x, t3])
    #t5s = Merge(mode='mul', concat_axis=-1)([x, t5])
    #tall = Merge(mode='mul', concat_axis=-1)([t1, t3, t5])
    x = Merge(mode='concat', concat_axis=-1)([x, t1, t3, t5])
    
    
    
    x = PReLU()(x)
    x = Dropout(0.35)(x)  #0.25
    #y = Dense(8, activation='softmax')(x)
    y = Dense(452, activation='sigmoid')(x)

    
    #model = Model(inputs=[in1, in2], outputs=y)
    model = Model(inputs=[in1], outputs=y)
    
    #print (model.summary())
    #rmsprop = RMSprop(lr=0.005, rho=0.9, epsilon=1e-08, decay=0.0)
    #rmsprop = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False) #0.430
    #rmsprop = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-06)
    rmsprop = keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #rmsprop = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    
    #rmsprop = keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-06) #lr=1.0 rho=0.95
    #model.compile(optimizer=rmsprop,  loss='categorical_crossentropy',   metrics=[Jaccard_Sim]) #binary_crossentropy
    model.compile(optimizer=rmsprop,  loss='binary_crossentropy',   metrics=[Jaccard_Sim]) #use1
    #model.compile(optimizer=rmsprop,  loss=jaccard_distance_loss,   metrics=[Jaccard_Sim])


    global seed
    #model_name = 'model/dnn.seed.'+str(seed)+'.h5'
    sbo = False
    if online == 1:
        sbo = True
        
        
    if ts > 0:
        model_check = ModelCheckpoint(filepath=model_path + 'seed.'+str(seed)+'.{val_Jaccard_Sim:.4f}.weights.{epoch:02d}.hdf5', save_best_only=sbo, verbose=1) #False
    else:
        model_check = ModelCheckpoint(filepath=model_path + 'seed.'+str(seed)+'.weights.fawen.zi.{epoch:02d}.hdf5', save_best_only=sbo, verbose=1) #False


    if len(test_x) == 0:
        flag = 0
        if flag == 1:
            model.fit(train_x, train_y, batch_size=32, epochs=epoch, shuffle=True, validation_split=0.2, callbacks=[model_check]) #16
        else:
            
            if ts > 0 :
                train_X, test_X, train_y, test_y = train_test_split(train_x[0], train_y, test_size = ts, random_state = seed)
    
                #model.load_weights('model/seed.99.0.7989.weights.04.hdf5')
                
                model.fit([train_X], train_y, batch_size=32, epochs=epoch, shuffle=True, validation_data = ([test_X], test_y), callbacks=[model_check])
                
    #            preds = model.predict([test_X], batch_size=32, verbose=0)
    #            sco = []
    #            for i in range(len(preds)):
    #                a = []
    #                b = []
    #                for j in range(len(test_y[0])):
    #                    if test_y[i][j] > 0:
    #                        a.append(j)
    #                for j in range(len(preds[0])):
    #                    if preds[i][j] > jac_thre:
    #                    #preds[i][j] = int(round(preds[i][j]))
    #                    #if preds[i][j] == 1:
    #                        b.append(j)
    #                a = set(a)
    #                b = set(b)
    #                sco.append(len(a&b)/len(a|b))
    #                #print (a, b)
    #            print ('Test Jac:', np.array(sco).mean())
            else:
                model.fit(train_x, train_y, batch_size=32, epochs=epoch, shuffle=True, callbacks=[model_check])
        del model
        gc.collect()                     
    else:
        #model.fit(train_x, train_y, batch_size=16, epochs=epoch, shuffle=True) #16
        global model_name
        model.load_weights(model_name)
        
        preds = model.predict(test_x, batch_size=32, verbose=0)
        #print (preds)
        del model
        gc.collect()        
        return preds
    

    
def load_feat(flag):
    if flag == 'train':
        df = pd.read_csv(train_path, header=None, encoding='utf8') #, nrows=1000 , nrows=10000
        #df = pd.read_csv('feature/train.tsv', header=None, encoding='utf8', nrows=1000) #, nrows=1000
    else:
        df = pd.read_csv(test_path, header=None, encoding='utf8') #, nrows=1000
        #df = pd.read_csv('feature/test.tsv', header=None, encoding='utf8', nrows=1000) #, nrows=1000
    
    #df = df.fillna(max_char_id+2)
    
    df = df.fillna(0)
    #df = df.astype('int')
    
    X = []
    Y = []
    if flag == 'train':
        train_x = df[df.columns[2:max_seq_len+2]].values
        y = df[1].values
        train_y = []
        for yi in y:
            t = [0]*452
            
            sp = yi.split(',')
            for pi in sp:
                t[int(pi)-1] = 1
            train_y.append(t)
            
        X, Y = [np.array(train_x)], np.array(train_y)
        #return [np.array(train_x)], np.array(train_y)
    else:
        test_x = df[df.columns[2:max_seq_len+2]].values
        test_id = df[0].values
        X, Y = [np.array(test_x)], test_id
        #return [test_x], test_id
        

    return X, Y


def save_res(ids, preds):
    res_path = result_path
    prob_path = res_path + '.prob.tsv'
    X = []
    T = []
    for i in range(len(ids)):
        x = [ids[i]]
        t = []
        
        T.append(x + list(preds[i]))
        
        for j in range(len(preds[i])):
            if preds[i][j] > 0.5:
                t.append(str(j+1))
        x.append(','.join(t))
        X.append(x)
        
    df = pd.DataFrame(X)
    df.to_csv(res_path, index=False, header=False)
    
    df = pd.DataFrame(T)
    df.to_csv(prob_path, index=False, header=False, float_format = '%.6f')
    


model_name = ''
online = 0
seed = 0
ts = 0.2

def run(mon):
    global seed, model_name, ts, online

    online = mon
    ts = 0
    
    print ('load data...')
    #train_x, train_y = load_train()
    train_x, train_y = load_feat('train')
    
    #0, 8, 77, 128 1024
    if online == 0:
        seeds = [1 ,4, 16, 64, 99, 128, 666, 999, 1225, 10101]
        #seeds = [4399]
        for seed in seeds:
            print ('use seed:', seed)
            np.random.seed(seed) #666
            print ('strat traing...')
            build_model(train_x, train_y)
            #break
    else:
        #test_x, test_id = load_test()
        test_x, test_id = load_feat('test')
        

        print ('strat traing...')
        preds = []
        #for i in range(len(models)):
        #path = 'model/10top3'
        path = model_path
        
        #cnts = []

        model_cnt = 0
        for dirpath,dirnames,filenames in os.walk(path):
            for file in filenames:
                mepo = int(file[-7:-5]) #选择3,4,5个epoch
                if mepo in [3,4,5] and file.find('fawen.zi') >= 0:   
                    
                    model_cnt += 1
                    fullpath = os.path.join(dirpath,file)
                    model_name = fullpath
                    #model_name = models[i]
                    print (model_name)
                    np.random.seed(seed)
                    predsi = build_model(train_x, train_y, test_x)
                    if len(preds) == 0:
                        preds = predsi #*weis[i]
                    else:
                        for i in range(len(preds)):
                            preds[i] += predsi[i] #*weis[i]
                        
                #break

                
        for i in range(len(preds)):
            preds[i] /= model_cnt
            
        print (preds)
        
        save_res(test_id, preds)
        
        
def main():
    global seed, model_name, ts, online

    online = 1
    ts = 0.1
    
    print ('load data...')
    #train_x, train_y = load_train()
    train_x, train_y = load_feat('train')
    
    #0, 8, 77, 128 1024
    if online == 0:
        seeds = [1 ,4, 16, 64, 99, 128, 666, 999, 1225, 10101]
        #seeds = [4399]
        for seed in seeds:
            print ('use seed:', seed)
            np.random.seed(seed) #666
            print ('strat traing...')
            build_model(train_x, train_y)
    else:
        #test_x, test_id = load_test()
        test_x, test_id = load_feat('test')
        

        print ('strat traing...')
        preds = []
        #for i in range(len(models)):
        path = 'model/10top3'
        #path = 'model/test'
        
        #cnts = []

        model_cnt = 0
        for dirpath,dirnames,filenames in os.walk(path):
            for file in filenames:
                model_cnt += 1
                fullpath = os.path.join(dirpath,file)
                model_name = fullpath
                #model_name = models[i]
                print (model_name)
                np.random.seed(seed)
                predsi = build_model(train_x, train_y, test_x)
                if len(preds) == 0:
                    preds = predsi #*weis[i]
                else:
                    for i in range(len(preds)):
                        preds[i] += predsi[i] #*weis[i]
                        
                #break

                
        for i in range(len(preds)):
            preds[i] /= model_cnt
            
        print (preds)
        
        save_res(test_id, preds)
    
if __name__ == '__main__':
    main()

    