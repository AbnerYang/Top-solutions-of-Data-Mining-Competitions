# -*- encoding:utf-8 -*-
from __future__ import print_function,unicode_literals,division
import os
import os.path as osp
import pickle
import numpy as np
import tensorflow as tf
from model_link20 import *


def get_train_model(data_set, interval, reuse, batchsize, shuffle, train, name='model'):
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        batch_link_ftr, batch_link_id, batch_link_g, \
        batch_route_ftr, batch_route_id, batch_route_g, \
        batch_wea_ftr, batch_time_ftr, \
        batch_times, batch_route_tgt, batch_weights = get_inputs(
            data_dir, data_set, interval=interval, batch_size=batchsize, shuffle=shuffle)
        inputs = batch_link_ftr, batch_link_id, batch_link_g, \
                 batch_route_ftr, batch_route_id, batch_route_g, \
                 batch_wea_ftr, batch_time_ftr

        route_preds, ftrs = inference(inputs, data, reuse, train, name='{}'.format(interval))
        net = tl.layers.ConcatLayer([route_preds[k] for k in sorted(route_preds)], concat_dim=-1, name='concat_all')
        target = tf.concat([batch_route_tgt[k] for k in sorted(route_preds)], axis=-1)
        loss = get_loss(target, net.outputs, batch_weights)
        return net, ftrs, loss


steps = 30000
freq = 100
# interval=40
intervals = [15, 20, 24]  # [5, 8, 10, 12, 15, 17, 20, 24, 30, 40][3:]
if __name__ == '__main__':
    data_dir = '../../data/dataSets'
    data = Data(data_dir)
    for interval in intervals:
        net, ftrs, loss = get_train_model('train', interval, False, 32, True,train=True, name='train')
        net_val, ftrs_val, loss_val = get_train_model('sval', interval, False, 2 * 7, False, train=False, name='sval')
        net_val2, ftrs_val2, loss_val2 = get_train_model('val', interval, False, 60 * 7, False, train=False, name='val')

        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(0.001, global_step, decay_steps=100, decay_rate=0.95)
        ema = tf.train.ExponentialMovingAverage(0.99)
        vars_avg_op = ema.apply(net.all_params)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss, var_list=net.all_params)
        train_op = tf.group(train_op, vars_avg_op)

        init = tf.global_variables_initializer()
        val_loss = 1.0
        save_path = 'models/{}_{}_final.npz'.format('train', interval)
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            losses = []
            for i in range(1, steps + 1):
                _, l = sess.run([train_op,  loss],feed_dict=net.all_drop)
                print('\r{:0>6} train loss:{:.4f}'.format(i, l), end='')
                losses.append(l)
                if i % freq == 0:
                    print('\r{:0>6} train_loss:{:.5f}'.format(i, np.mean(losses)), end='\t')
                    losses = []
                    save_list_var = sess.run([ema.average(x) for x in net.all_params])
                    tl.files.assign_params(sess, save_list_var, net_val)
                    v_loss = sess.run(loss_val, feed_dict=tl.utils.dict_to_one(net_val.all_drop))
                    # v_loss=0

                    tl.files.assign_params(sess, save_list_var, net_val2)
                    v2_loss = sess.run(loss_val2, feed_dict=tl.utils.dict_to_one(net_val2.all_drop))

                    print('sval_loss: {:.4f}, val_loss: {:.4f}'.format(v_loss,v2_loss), end=' \t')
                    if v2_loss < val_loss:
                        val_loss = v2_loss
                        # save model:
                        # tl.files.save_npz([ema.average(x) for x in net.all_params], save_path)
                    else:
                        print('not improved!')
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        sess.close()
        tf.reset_default_graph()
        tl.layers.clear_layers_name()
