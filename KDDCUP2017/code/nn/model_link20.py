# -*- encoding:utf-8 -*-
from __future__ import print_function, unicode_literals, division
import os
import datetime
import os.path as osp
import pickle
import numpy as np
import tensorflow as tf

np.random.seed(0)
tf.set_random_seed(1234)
import tensorlayer as tl
from preprocessing import Data


def link_inference_cnn(link_in, link_id, link_g, train, reuse, name):
    T, dim = [int(i) for i in link_in.get_shape()[1:]]
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(link_in, name=name +'input')
        net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, dim]), name=name +'lambda1')
        net = tl.layers.DenseLayer(net, n_units=8, W_init=tf.contrib.layers.xavier_initializer(),name=name +'share_dense')
        net = tl.layers.LambdaLayer(net, lambda x: tf.reshape(x, [-1, T, 8, 1]), name=name +'lambda2')
        # net=tl.layers.LambdaLayer(net,lambda x:tf.image.resize_bilinear(x, [8,8]),name=name+'lambda3')
        net = tl.layers.Conv2dLayer(net, shape=[3, 3, 1, 16], W_init=tf.contrib.layers.xavier_initializer(),
                                    name=name +'conv1')
        net = tl.layers.PoolLayer(net, name=name +'pool1')
        net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name=name +'prelu1')
        net = tl.layers.Conv2dLayer(net, shape=[2, 2, 16, 32], padding='VALID',
                                    W_init=tf.contrib.layers.xavier_initializer(), name=name +'conv2')
        net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name=name + 'prelu2')
        net = tl.layers.FlattenLayer(net, name=name +'flatten')
        net = tl.layers.DropoutLayer(net, 0.5, is_fix=True, is_train=train, name=name + 'dorp1')
        net = tl.layers.DenseLayer(net, n_units=48, W_init=tf.contrib.layers.xavier_initializer(),
                                   act=tf.identity, name=name +'dense1')
        return net


def link_inference_rnn(link_in, link_id, link_g, train, reuse, name):
    batch_size, T, dim = [int(i) for i in link_in.get_shape()]
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(link_in, name=name + 'input')

        g = tl.layers.InputLayer(link_g, name=name + 'g_in')
        g = tl.layers.DenseLayer(g, 16, name=name + 'g_dense')

        link_embed = tl.layers.EmbeddingInputlayer(link_id, vocabulary_size=24,
                                                   embedding_size=10, name=name + 'embed')
        link_embed = tl.layers.LambdaLayer(link_embed, lambda x: tf.squeeze(x), name=name + 'lambed1')
        # link_embed = tl.layers.DenseLayer(link_embed, 10, name=name + 'embed_dense')
        # link_embed = tl.layers.PReluLayer(link_embed, a_init=tf.constant_initializer(0.1), name=name + 'embed__prelu')
        # net = tl.layers.ConcatLayer([net, link_embed], concat_dim=-1, name=name + 'concat')
        # net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.GRUCell,
        #                          cell_init_args={'activation': tf.nn.tanh}, n_hidden=32,
        #                          n_steps=T, return_last=False,
        #                          name=name + 'rnn0')
        net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.GRUCell,
                                 cell_init_args={'activation': tf.identity}, n_hidden=64,
                                 n_steps=T, return_last=True,
                                 name=name + 'rnn2')
        net = tl.layers.ConcatLayer([net, g, link_embed], concat_dim=-1, name=name + 'concat2')
        # net=tl.layers.DenseLayer(net,64,name=name+'dense2')
        # net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name=name + 'prelu2')
        return net


def route_inference_rnn(route_in, route_id, route_g, reuse, name):
    T, dim = [int(i) for i in route_in.get_shape()[1:]]
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = tl.layers.InputLayer(route_in, name=name + 'input')

        g = tl.layers.InputLayer(route_g, name=name + 'g_in')
        g = tl.layers.DenseLayer(g, 16, name=name + 'g_dense')

        route_embed = tl.layers.EmbeddingInputlayer(route_id, vocabulary_size=6,
                                                    embedding_size=5, name=name + 'embed')
        route_embed = tl.layers.LambdaLayer(route_embed, lambda x: tf.squeeze(x), name=name + 'lambed1')
        # net = tl.layers.ConcatLayer([net, route_embed], concat_dim=-1, name=name + 'concat')
        # net=tl.layers.ConcatLayer([net0,net],name=name+'concat1')
        # net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.GRUCell,
        #                          cell_init_args={'activation': tf.nn.tanh}, n_hidden=32,
        #                          n_steps=T, return_last=False,
        #                          name=name + 'rnn0')
        net = tl.layers.RNNLayer(net, cell_fn=tf.contrib.rnn.GRUCell,
                                 cell_init_args={'activation': tf.identity}, n_hidden=32,
                                 n_steps=T, return_last=True,
                                 name=name + 'rnn2')
        net=tl.layers.ConcatLayer([net,g,route_embed],name=name+'concat2')
        return net


def time_inference(time_in, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net=tl.layers.InputLayer(time_in,name=name+'in')
        # net=tl.layers.DenseLayer(net, 16, act=tf.nn.relu, name=name+'dense1')
        net = tl.layers.DenseLayer(net, 8, act=tf.nn.relu, name=name + 'out')
    return net


def wea_inference(time_in, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net=tl.layers.InputLayer(time_in, name=name+'in')
        # net=tl.layers.DenseLayer(net, 16, act=tf.nn.relu, name=name+'dense1')
        net = tl.layers.DenseLayer(net, 10, act=tf.nn.relu, name=name + 'out')
    return net


def mape_loss(y_true, y_pred, weights):
    partition = tf.cast(tf.less(y_true, 1), tf.int32)
    yy_true, _ = tf.dynamic_partition(y_true, partition, 2)
    yy_pred, _ = tf.dynamic_partition(y_pred, partition, 2)
    weights, _ = tf.dynamic_partition(weights, partition, 2)
    loss = tf.divide(tf.abs(yy_true - yy_pred), yy_true) * weights
    loss = tf.reduce_mean(loss)
    return loss


def links_to_route(link_ftrs, route_ftrs, wea_ftrs, time_ftrs,  bias, train, reuse, name):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        wea_ftrs = time_inference(wea_ftrs, reuse=reuse, name='wearef')
        time_ftrs=time_inference(time_ftrs,reuse=reuse,name='timeref')
        net = tl.layers.ElementwiseLayer(link_ftrs, tf.add, name=name + 'elem')

        net=tl.layers.ConcatLayer([net, time_ftrs, wea_ftrs, route_ftrs],name=name+'concat1')

        net = tl.layers.DropoutLayer(net, 0.5, is_fix=False, is_train=train, name=name + 'dorp1')
        net = tl.layers.DenseLayer(net, 128, name=name + 'dense1')
        net = tl.layers.LambdaLayer(net,lambda x:tf.nn.relu(x), name=name+'lambda1')
        # net = tl.layers.DropoutLayer(net, 0.5, is_fix=False, is_train=train, name=name + 'dorp2')
        ftr = net
        net = tl.layers.DenseLayer(net, 128, name=name + 'dense2')
        net = tl.layers.PReluLayer(net, a_init=tf.constant_initializer(0.1), name=name + 'prelu2')
        # net = tl.layers.DropoutLayer(net, 0.5, is_fix=False, is_train=train, name=name + 'dorp3')
        net = tl.layers.DenseLayer(net, 6, b_init=tf.constant_initializer(value=0), name=name + 'out')
        net = tl.layers.LambdaLayer(net, lambda x: x + bias, name=name + 'lam')
        return net,ftr


def inference(inputs, data, reuse, train, name=''):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        batch_link_ftr, batch_link_id, batch_link_g, \
        batch_route_ftr, batch_route_id, batch_route_g, \
        batch_wea_ftr, batch_time_ftr = inputs

        link_ids = sorted(batch_link_ftr.keys())
        route_ids = sorted(batch_route_ftr.keys())

        # link inference
        link_ftrs = {}
        for i, k in enumerate(link_ids):
            ru = False if (i == 0 and not reuse) else True
            link_ftr = link_inference_rnn(batch_link_ftr[k], batch_link_id[k], batch_link_g[k], train,
                                          reuse=ru, name=name + 'linkrnn')
            link_cnn_ftr=link_inference_cnn(batch_link_ftr[k], batch_link_id[k], batch_link_g[k], train,
                                          reuse=ru, name=name + 'linkcnn')
            link_ftr=tl.layers.ConcatLayer([link_ftr,link_cnn_ftr],name=name+'linkconcat')
            link_ftrs[k] = link_ftr

        # route inference pred
        bias = [68.891449, 116.24596, 76.759598, 105.80603, 122.24135, 107.88319]
        route_preds = {}
        ftrs={}
        for i, k in enumerate(route_ids):
            ru = False if (i == 0 and not reuse) else True
            # ru = False if not reuse else True
            route_link_ftrs = [link_ftrs[k] for k in data.route[k]]
            route_ftr = route_inference_rnn(batch_route_ftr[k], batch_route_id[k], batch_route_g[k],
                                            reuse=ru, name=name + 'routernn')
            route_pred,gftr = links_to_route(route_link_ftrs, route_ftr, batch_wea_ftr, batch_time_ftr, bias[i], train, reuse=ru, name=name + 'routepred')
            route_preds[k] = route_pred
            ftrs[k] = gftr
    return route_preds, ftrs


def get_loss(y_true, y_pred, weights):
    loss = mape_loss(y_true, y_pred, weights)
    return loss


def global_ftr(data, nb_sample):
    gdata = {}
    for k in data:
        ftr = []
        for i in range(nb_sample):
            ft = []
            for j in [0,2]:
                d = data[k][i, :, j]
                ft.extend([np.mean(d), np.median(d),
                           np.max(d),
                           np.min(d),
                           np.std(d),
                           np.mean(d) - np.std(d),
                           np.mean(d) + np.std(d),
                           np.max(d) - np.min(d)])
                # ft.extend(d.tolist())
                # ft.extend(np.diff(d).tolist())
            ftr.append(ft)
        ftr = np.array(ftr, dtype='float32')
        gdata[k] = ftr
    return gdata


def get_inputs(data_dir, data_set='train', interval=20, batch_size=16, shuffle=True, seed=2017):
    with open('../../feature/nn/{}_ftrs_{}.pkl'.format(data_set, interval), 'rb') as f:
            route_ftrs, link_ftrs, wea_ftrs, times_ftrs, _,_,_,_ = pickle.load(f)
    with open('../../feature/nn/{}_ftrs_{}.pkl'.format(data_set, 20), 'rb') as f:
            _, _, _, _, tms, route_targets, link_targets, deltas = pickle.load(f)
            weights = np.exp(-deltas ** 2 / (2 * 35 ** 2))
    with open('../../feature/nn/{}_ftrs_{}.pkl'.format(data_set, 120), 'rb') as f:
            route_ftrs2, link_ftrs2, _,_,_,_,_,_ = pickle.load(f)
    # if data_set=='train':
    #     idxs=[x<str(datetime.datetime(2016, 10, 11)) for x in tms]
    # elif data_set=='val':
    #     idxs=[str(datetime.datetime(2016, 10, 11)) < x < str(datetime.datetime(2016, 10, 18)) for x in tms]
    #     weights = np.ones_like(weights, dtype='float32')
    # elif data_set=='sval':
    #     idxs=[x in [str(datetime.datetime(2016, 10, day, hour)) for day in range(11, 18) for hour in [8, 17]] for x in tms]
    #     weights = np.ones_like(weights, dtype='float32')
    # else:
    #     idxs=[x>str(datetime.datetime(2016, 10, 18)) for x in tms]
    #     weights = np.ones_like(weights, dtype='float32')
    #
    # idxs=np.where(idxs)[0]
    # route_ftrs = {k: v[idxs] for k, v in route_ftrs.items()}
    # link_ftrs = {k: v[idxs] for k, v in link_ftrs.items()}
    # route_targets = {k: v[idxs] for k, v in route_targets.items()}
    # # link_targets = {k: v[idxs] for k, v in link_targets.items()}
    # wea_ftrs=wea_ftrs[idxs]
    # tms=[tms[x] for x in idxs.tolist()]
    # times_ftrs=times_ftrs[idxs]
    # route_ftrs2 = {k: v[idxs] for k, v in route_ftrs2.items()}
    # link_ftrs2 = {k: v[idxs] for k, v in link_ftrs2.items()}
    #
    # weights = weights[idxs]
    if not data_set.endswith('train'):
        weights = np.ones_like(weights, dtype='float32')
    weights = weights.reshape((-1, 1))
    weights = np.repeat(weights, 36, axis=1)

    nb_sample, T = route_ftrs['B1'].shape[:2]
    link_ids = sorted(link_ftrs.keys())
    route_ids = sorted(route_ftrs.keys())

    # 计算全局特征，合并interval=120的特征
    route_ftrs2 = {k: np.reshape(v, (nb_sample, -1)) for k, v in route_ftrs2.items()}
    link_ftrs2 = {k: np.reshape(v, (nb_sample, -1)) for k, v in link_ftrs2.items()}

    glink_ftrs = global_ftr(link_ftrs, nb_sample)
    groute_ftrs = global_ftr(route_ftrs, nb_sample)

    glink_ftrs = {k: np.concatenate((v, link_ftrs2[k]), axis=-1)
                  for k, v in glink_ftrs.items()}
    groute_ftrs = {k: np.concatenate((v, route_ftrs2[k]), axis=-1)
                   for k, v in groute_ftrs.items()}

    if interval == 20:
        ftr_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        ftr_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    link_ftr_inputs = {}
    link_id_inputs = {}
    link_g_inputs = {}
    for i, k in enumerate(link_ids):
        link_ftr_inputs[k] = tf.constant(np.array(link_ftrs[k][:, :, :], dtype='float32'))
        link_id_inputs[k] = tf.zeros((nb_sample, 1), 'int32') + i
        link_g_inputs[k] = tf.constant(np.array(glink_ftrs[k], dtype='float32'))

    route_ftr_inputs = {}
    route_id_inputs = {}
    route_g_inputs = {}
    route_tgt_inputs={}
    for i, k in enumerate(route_ids):
        route_ftr_inputs[k] = tf.constant(np.array(route_ftrs[k][:, :, :], dtype='float32'))
        route_id_inputs[k] = tf.zeros((nb_sample, 1), 'int32') + i
        route_g_inputs[k] = tf.constant(np.array(groute_ftrs[k], dtype='float32'))
        route_tgt_inputs[k] = tf.constant(np.array(route_targets[k], dtype='float32'))

    weights_input = tf.constant(weights, dtype='float32')
    wea_ftr_inputs = tf.constant(wea_ftrs, dtype='float32')
    time_ftr_inputs = tf.constant(times_ftrs, dtype='float32')
    tms = tf.constant(tms, dtype=tf.string)
    cap = nb_sample
    mad = nb_sample // 4
    if shuffle:
        batch_link_ftr = tf.train.shuffle_batch(link_ftr_inputs, batch_size=batch_size, capacity=cap,
                                                min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_link_id = tf.train.shuffle_batch(link_id_inputs, batch_size=batch_size, capacity=cap,
                                               min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_link_g = tf.train.shuffle_batch(link_g_inputs, batch_size=batch_size, capacity=cap,
                                              min_after_dequeue=mad, enqueue_many=True, seed=seed)

        batch_route_ftr = tf.train.shuffle_batch(route_ftr_inputs, batch_size=batch_size, capacity=cap,
                                                 min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_route_id = tf.train.shuffle_batch(route_id_inputs, batch_size=batch_size, capacity=cap,
                                                min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_route_g = tf.train.shuffle_batch(route_g_inputs, batch_size=batch_size, capacity=cap,
                                               min_after_dequeue=mad, enqueue_many=True, seed=seed)

        batch_wea_ftr = tf.train.shuffle_batch([wea_ftr_inputs], batch_size=batch_size, capacity=cap,
                                               min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_time_ftr = tf.train.shuffle_batch([time_ftr_inputs], batch_size=batch_size, capacity=cap,
                                                min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_time = tf.train.shuffle_batch([tms], batch_size=batch_size, capacity=cap,
                                            min_after_dequeue=mad, enqueue_many=True, seed=seed)

        batch_route_tgt = tf.train.shuffle_batch(route_tgt_inputs, batch_size=batch_size, capacity=cap,
                                                 min_after_dequeue=mad, enqueue_many=True, seed=seed)
        batch_weights = tf.train.shuffle_batch([weights_input], batch_size=batch_size, capacity=cap,
                                               min_after_dequeue=mad, enqueue_many=True, seed=seed)

    else:
        batch_link_ftr = tf.train.batch(link_ftr_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_link_id = tf.train.batch(link_id_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_link_g = tf.train.batch(link_g_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)

        batch_route_ftr = tf.train.batch(route_ftr_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_route_id = tf.train.batch(route_id_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_route_g = tf.train.batch(route_g_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)

        batch_wea_ftr = tf.train.batch([wea_ftr_inputs], batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_time_ftr = tf.train.batch([time_ftr_inputs], batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_time = tf.train.batch([tms], batch_size=batch_size, capacity=200, enqueue_many=True)

        batch_route_tgt = tf.train.batch(route_tgt_inputs, batch_size=batch_size, capacity=200, enqueue_many=True)
        batch_weights = tf.train.batch([weights_input], batch_size=batch_size, capacity=200, enqueue_many=True)
    return batch_link_ftr, batch_link_id, batch_link_g, \
           batch_route_ftr, batch_route_id, batch_route_g, \
           batch_wea_ftr, batch_time_ftr, batch_time, \
           batch_route_tgt, batch_weights
