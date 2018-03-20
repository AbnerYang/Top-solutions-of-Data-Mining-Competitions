from __future__ import unicode_literals

import h5py
import codecs
import datetime
import numpy as np
import data_input

from  keras.layers import Dense, Dropout, GRU, BatchNormalization, \
    Embedding, Input, TimeDistributed, Merge, PReLU, Lambda
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras import backend as K

np.random.seed(0)
prefix = '/home/lllcho/Documents/season_1/'
train_prefix = prefix + 'training_data/'
test_prefix = prefix + 'test_set_1/'
predict_idx = range(46, 144, 12)
used_idx = [i - j for j in range(1, 4) for i in predict_idx]
used_idx.sort()
used_idx_map = dict([(v, i) for i, v in enumerate(used_idx)])
# cluster2id, id2cluster = data_input.get_cluster_map(test_prefix + 'cluster_map/cluster_map')

with h5py.File('train_data.h5', 'r') as f:
    train_orders_dest = f['dest'][:]
    train_orders_price = f['price'][:]
    train_orders_gap = f['gap'][:]
    train_weathers = f['weathers'][:]
    train_traffics = f['traffics'][:]
    train_orders_dest = train_orders_dest[:, :, :, 1:]
    train_orders_price = train_orders_price[:, :, :, 1:]
    train_orders_come = train_orders_dest.transpose((0, 1, 3, 2))
    train_weathers_weather = train_weathers[:, :, 0].astype('int8')

with h5py.File('test_data.h5', 'r') as f:
    test_orders_dest = f['dest'][:]
    test_orders_price = f['price'][:]
    test_orders_gap = f['gap'][:]
    test_weathers = f['weathers'][:]
    test_traffics = f['traffics'][:]
    test_orders_dest = test_orders_dest[:, :, :, 1:]
    test_orders_price = test_orders_price[:, :, :, 1:]
    test_orders_come = test_orders_dest.transpose((0, 1, 3, 2))
    test_weathers_weather = test_weathers[:, :, 0].astype('int8')

test_orders_price = np.nan_to_num(test_orders_price / test_orders_dest)
test_orders_price[test_orders_price > 300] = 300
test_orders_price = test_orders_price / test_orders_price.max()
test_orders_dest /= 300

test_weathers = test_weathers / train_weathers.max(axis=(0, 1))
test_weathers = np.repeat(np.expand_dims(test_weathers, 2), 66, axis=2)

test_traffics = test_traffics / np.expand_dims(test_traffics.sum(axis=3), axis=3)
test_traffics = np.nan_to_num(test_traffics)
test_traffics[:, :, 53] = test_traffics.mean(axis=2)

x_test_gap = np.zeros((5, 9, 66, 3), dtype='float32')
x_test_come = np.zeros((5, 9, 66, 3, 66), dtype='float32')
x_test_price = np.zeros((5, 9, 66, 3, 66), dtype='float32')
x_test_weather = np.zeros((5, 9, 66, 3, 3), dtype='float32')
x_test_weather_weather = np.zeros((5, 9, 66, 3), dtype='int32')
x_test_traffic = np.zeros((5, 9, 66, 3, 4), dtype='float32')
x_test_pio = np.zeros((5, 9, 66, 3), dtype='int32')
x_test_weekday = np.zeros((5, 9, 66, 3), dtype='int32')
x_test_time = np.zeros((5, 9, 66, 3), dtype='int32')
x_test_dest = np.zeros((5, 9, 66, 3, 66), dtype='float32')
for id, date in enumerate([22, 24, 26, 28, 30]):
    for it, time_idx in enumerate([p - 1 for p in predict_idx]):
        for dist_idx in range(66):
            x_test_gap[id, it, dist_idx] = test_orders_gap[id, time_idx - 3:time_idx, dist_idx]
            x_test_come[id, it, dist_idx] = test_orders_come[id, time_idx - 3:time_idx, dist_idx]
            x_test_price[id, it, dist_idx] = test_orders_price[id, time_idx - 3:time_idx, dist_idx]
            x_test_weather[id, it, dist_idx] = test_weathers[id, time_idx - 3:time_idx, dist_idx]
            x_test_weather_weather[id, it, dist_idx] = test_weathers_weather[id, time_idx - 3:time_idx] - 1
            x_test_traffic[id, it, dist_idx] = test_traffics[id, time_idx - 3:time_idx, dist_idx]
            x_test_pio[id, it, dist_idx] = dist_idx
            x_test_weekday[id, it, dist_idx] = datetime.date(2016, 1, date).weekday()
            x_test_time[id, it, dist_idx] = np.asarray([(time_idx - 44 - i) / 2 for i in [3, 2, 1]])
            x_test_dest[id, it, dist_idx] = test_orders_dest[id, time_idx - 3:time_idx, dist_idx]

x_test_gap = x_test_gap.reshape((np.prod(x_test_gap.shape[:3]), 3, 1))
x_test_come = x_test_come.reshape((np.prod(x_test_come.shape[:3]), 3, 66))
x_test_price = x_test_price.reshape((np.prod(x_test_price.shape[:3]), 3, 66))
x_test_weather = x_test_weather.reshape((np.prod(x_test_weather.shape[:3]), 3, 3))
x_test_weather_weather = x_test_weather_weather.reshape((np.prod(x_test_weather_weather.shape[:3]), 3))
x_test_traffic = x_test_traffic.reshape((np.prod(x_test_traffic.shape[:3]), 3, 4))
x_test_pio = x_test_pio.reshape((np.prod(x_test_pio.shape[:3]), 3))
x_test_weekday = x_test_weekday.reshape((np.prod(x_test_weekday.shape[:3]), 3))
x_test_time = x_test_time.reshape((np.prod(x_test_time.shape[:3]), 3))
x_test_dest = x_test_dest.reshape((np.prod(x_test_dest.shape[:3]), 3, 66))
x_test_gap /= 10

train_orders_price = np.nan_to_num(train_orders_price / train_orders_dest)
train_orders_price[train_orders_price > 300] = 300
train_orders_price = train_orders_price / train_orders_price.max()
train_orders_dest /= 300

train_weathers = train_weathers / train_weathers.max(axis=(0, 1))
train_weathers = np.repeat(np.expand_dims(train_weathers, 2), 66, axis=2)

train_traffics = train_traffics / np.expand_dims(train_traffics.sum(axis=3), axis=3)
train_traffics = np.nan_to_num(train_traffics)
train_traffics[:, :, 53] = train_traffics.mean(axis=2)

x_train_gap = np.zeros((21, 100, 66, 3), dtype='float32')
y_train_gap = np.zeros((21, 100, 66,), dtype='float32')
x_train_come = np.zeros((21, 100, 66, 3, 66), dtype='float32')
x_train_price = np.zeros((21, 100, 66, 3, 66), dtype='float32')
x_train_weather = np.zeros((21, 100, 66, 3, 3), dtype='float32')
x_train_weather_weather = np.zeros((21, 100, 66, 3), dtype='int32')
x_train_traffic = np.zeros((21, 100, 66, 3, 4), dtype='float32')
x_train_pio = np.zeros((21, 100, 66, 3), dtype='int32')
x_train_weekday = np.zeros((21, 100, 66, 3), dtype='int32')
x_train_time = np.zeros((21, 100, 66, 3), dtype='int32')
weight_idx = np.ones((21, 100, 66), dtype='float32')
x_train_dest = np.zeros((21, 100, 66, 3, 66), dtype='float32')

for id, date in enumerate(range(1, 22)):
    for it, time_idx in enumerate(range(44, 144)):
        for dist_idx in range(66):
            x_train_gap[id, it, dist_idx] = train_orders_gap[id, time_idx - 3:time_idx, dist_idx]
            y_train_gap[id, it, dist_idx] = train_orders_gap[id, time_idx, dist_idx]
            x_train_come[id, it, dist_idx] = train_orders_come[id, time_idx - 3:time_idx, dist_idx]
            x_train_price[id, it, dist_idx] = train_orders_price[id, time_idx - 3:time_idx, dist_idx]
            x_train_weather[id, it, dist_idx] = train_weathers[id, time_idx - 3:time_idx, dist_idx] - 1
            x_train_weather_weather[id, it, dist_idx] = train_weathers_weather[id, time_idx - 3:time_idx] - 1
            x_train_traffic[id, it, dist_idx] = train_traffics[id, time_idx - 3:time_idx, dist_idx]
            x_train_pio[id, it, dist_idx] = dist_idx
            x_train_weekday[id, it, dist_idx] = datetime.date(2016, 1, date).weekday()
            x_train_time[id, it, dist_idx] = np.asarray([time_idx - 44 + i for i in [0, 1, 2]])
            x_train_dest[id, it, dist_idx] = train_orders_dest[id, time_idx - 3:time_idx, dist_idx]
            if time_idx in [p - 1 for p in predict_idx]:
                weight_idx[id, it] = 1

x_train_gap = x_train_gap.reshape((np.prod(x_train_gap.shape[:3]), 3, 1))
x_train_come = x_train_come.reshape((np.prod(x_train_come.shape[:3]), 3, 66))
x_train_price = x_train_price.reshape((np.prod(x_train_price.shape[:3]), 3, 66))
x_train_weather = x_train_weather.reshape((np.prod(x_train_weather.shape[:3]), 3, 3))
x_train_weather_weather = x_train_weather_weather.reshape((np.prod(x_train_weather_weather.shape[:3]), 3))
x_train_traffic = x_train_traffic.reshape((np.prod(x_train_traffic.shape[:3]), 3, 4))
x_train_pio = x_train_pio.reshape((np.prod(x_train_pio.shape[:3]), 3))
x_train_weekday = x_train_weekday.reshape((np.prod(x_train_weekday.shape[:3]), 3))
x_train_time = x_train_time.reshape((np.prod(x_train_time.shape[:3]), 3))
x_train_dest = x_train_dest.reshape((np.prod(x_train_dest.shape[:3]), 3, 66))
weight_idx = weight_idx.reshape((np.prod(weight_idx.shape),))
y_train_gap = y_train_gap.reshape((np.prod(y_train_gap.shape),))
index = y_train_gap > 0
x_train_gap /= 10

x_test = np.concatenate((x_test_gap,  # 1
                         x_test_weather,  # 3
                         x_test_traffic,  # 4
                         ), axis=-1)
x_train = np.concatenate((x_train_gap,  # 1
                          x_train_weather,  # 3
                          x_train_traffic,  # 4
                          ), axis=-1)
X = [x[index] for x in
     [x_train, x_train_pio, x_train_time, x_train_weekday, x_train_weather_weather, x_train_dest, x_train_come,
      x_train_price]]
Y = y_train_gap[index]

X_test = [x_test, x_test_pio, x_test_time, x_test_weekday, x_test_weather_weather, x_test_dest, x_test_come,
          x_test_price]

index_ = np.logical_not(index)
X_ = [x[index_] for x in
      [x_train, x_train_pio, x_train_time, x_train_weekday, x_train_weather_weather, x_train_dest, x_train_come,
       x_train_price]]
Y_ = y_train_gap[index_]
# Y_ += 1

idxs = np.random.permutation(index.sum())
split_at = int(index.sum() * 0.9)
x_val = [x[idxs[split_at:]] for x in X]
y_val = Y[idxs[split_at:]]
X_train = [x[idxs[:split_at]] for x in X]
Y_train = Y[idxs[:split_at]]
w = np.ones_like(Y_train)

# w2=np.ones_like(Y_)*0.1
# w=np.concatenate((w,w2))
# X_train =[np.concatenate(x,axis=0) for x in zip(X_train,X_)]
# Y_train = np.concatenate((Y_train,Y_))

T = 3
data = Input(shape=(T, 8))  ###
model = Sequential()
model.add(Dense(10, input_shape=(8,)))
model.add(PReLU())
datas = TimeDistributed(model)(data)

pio = Input(shape=(T,), dtype='int32')
pios = Embedding(input_dim=66, output_dim=5, input_length=T)(pio)

time = Input(shape=(T,), dtype='int32')
times = Embedding(input_dim=102, output_dim=5, input_length=T)(time)

week = Input(shape=(T,), dtype='int32')
weeks = Embedding(input_dim=7, output_dim=5, input_length=T)(week)

weather = Input(shape=(T,), dtype='int32')
weathers = Embedding(input_dim=9, output_dim=5, input_length=T)(weather)

dest = Input(shape=(T, 66))
model = Sequential()
model.add(Dense(5, input_shape=(66,)))
model.add(PReLU())
dests = TimeDistributed(model)(dest)

come = Input(shape=(T, 66))
model = Sequential()
model.add(Dense(5, input_shape=(66,)))
model.add(PReLU())
comes = TimeDistributed(model)(come)

price = Input(shape=(T, 66))  ###
model = Sequential()
model.add(Dense(5, input_shape=(66,)))
model.add(PReLU())
prices = TimeDistributed(model)(price)
data_all = Merge(mode='concat', concat_axis=-1)([datas, pios, times, weeks, weathers, dests, comes, prices])
data_all = BatchNormalization()(data_all)


def outshape(input_shape):
    shape = list(input_shape)
    shape[-2] -= 1
    return tuple(shape)


data_all2 = Lambda(lambda x: x[:, -2:, :], output_shape=outshape)(data_all)

gru2 = GRU(32, dropout_W=0.2, dropout_U=0.1)(data_all2)
gru3 = GRU(32, dropout_W=0.1, dropout_U=0.1)(data_all)

gru = Merge(mode='concat')([gru2, gru3])
x = Dropout(0.5)(gru)
x = Dense(32, W_regularizer=l2(1e-4))(x)
x = PReLU()(x)
x = Dropout(0.5)(x)
y = Dense(1, activation='relu')(x)
model = Model(input=[data, pio, time, week, weather, dest, come, price], output=y)

# with h5py.File('model/gru.h5', mode='r') as f:
#     flattened_layers = model.layers
#     layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
#     weight_value_tuples = []
#     for k, name in enumerate(layer_names[:]):
#         g = f[name]
#         weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
#         if len(weight_names):
#             weight_values = [g[weight_name] for weight_name in weight_names]
#             layer = flattened_layers[k]
#             symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
#             weight_value_tuples += zip(symbolic_weights, weight_values)
#     K.batch_set_value(weight_value_tuples)

model_name = 'model/gru.h5'
# s = model.to_json()
# with codecs.open(model_name + '.json', 'w') as f:
#     f.write(s)
model_check = ModelCheckpoint(model_name, save_best_only=True, verbose=1)


def get_lr(epoch):
    if epoch < 50:
        return 0.8e-3
    elif epoch < 100:
        return 5e-4
    elif epoch < 300:
        return 1e-4
    else:
        return 5e-5


lr = LearningRateScheduler(get_lr)
rms = RMSprop(lr=1e-5)
print ('compiling...')
model.compile(optimizer=rms, loss='mape', metrics=[])
model.fit(X_train, Y_train, batch_size=128, nb_epoch=100, sample_weight=None,
          validation_data=(x_val, y_val), callbacks=[model_check, lr])


# model.load_weights(model_name)
# pred = test_pred.reshape((5, 9, 66))
# with codecs.open('result/test_3937.csv', 'w') as f:
#     for id, date in enumerate([22, 24, 26, 28, 30]):
#         for it, time_idx in enumerate(predict_idx):
#             if (date, time_idx) in [(24, 46), (28, 46)]:
#                 pass
#             else:
#                 for dist_idx in range(66):
#                     f.write('{0},2016-01-{1}-{2},{3:.15f}\n'.format(
#                         dist_idx + 1, date, time_idx, pred[id, it, dist_idx]))
