# -*- encoding:utf-8 -*-
from __future__ import print_function,unicode_literals,division
import time
import datetime
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os.path as osp
from functools import reduce
import pandas as pd
from collections import defaultdict
from preprocessing import Data
from model import *

np.random.seed(0)


def get_path(path):
    while True:
        if osp.exists(path):
            time.sleep(1)
            return path
        else:
            time.sleep(1)


def mape_loss(y_true,y_pred):
    # idx=np.where(y_true>1)[0]
    y_pred=y_pred[y_true>1]
    y_true=y_true[y_true>1]
    mape=np.mean(np.abs(y_true-y_pred)/y_true)
    return mape

def get_ftrs(name, all_ftrs, all_tms):
    file_name='../../result/' + name+'.csv'
    times=[]
    intersections=[]
    tollgates=[]
    data=[]
    for k, v in all_ftrs.items():
        times.extend([str(t, encoding='utf-8') for t in all_tms])
        intersections.extend([k[0]]*len(all_tms))
        tollgates.extend([k[1]]*len(all_tms))
        data.append(v)
    data=np.concatenate(data)

    df=pd.DataFrame(data=list(zip(*[times,intersections,tollgates])), columns=['datetime','intersection_id','tollgate_id'])
    df2=pd.DataFrame(data=data,columns=['x{}'.format(i) for i in range(data.shape[1])],dtype='float32')
    df=pd.concat((df, df2),axis=1)
    df.to_csv(file_name, index=False, float_format='%g')
    # with open(pkl_name,'wb') as f:
    #     pickle.dump(df,f)


def get_submission(preds, name):
    with open('../../result/' + name, 'w') as f:
        f.write('intersection_id,tollgate_id,time_window,avg_travel_time\n')
        for it in ['A2', 'A3', 'B1', 'B3', 'C1', 'C3']:
            for i, t in enumerate(preds['time']):
                t = str(t, encoding='utf-8')
                t = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
                for im in range(6):
                    tm = t + datetime.timedelta(minutes=20)
                    line = '{},{},"[{},{})",{:.2f}\n'.format(it[0], it[1], str(t), str(tm), preds[it][i, im])
                    f.write(line)
                    t = tm

intervals=[24]
if __name__ == '__main__':
    test_intervals=[15,20,24]
    data_dir = '../../data/dataSets'
    data_set = 'test'  # sval
    data = Data(data_dir)
    if data_set != 'test':
        if data_set =='sval':
            batch_size=2*7
        elif data_set =='val':
            batch_size=60*7
        else:
            batch_size=3920

        all_preds=[]
        all_losses=[]
        for interval in intervals:
            batch_link_ftr, batch_link_id, batch_link_g, \
            batch_route_ftr, batch_route_id, batch_route_g, \
            batch_wea_ftr, batch_time_ftr,\
            batch_times,batch_route_tgt, batch_weights = get_inputs(
                data_dir, data_set, interval, batch_size=batch_size,shuffle=False)
            inputs=batch_link_ftr, batch_link_id, batch_link_g, \
            batch_route_ftr, batch_route_id, batch_route_g, \
            batch_wea_ftr, batch_time_ftr

            route_preds, ftrs = inference(inputs, data, False, False, name='{}'.format(interval))
            net = tl.layers.ConcatLayer([route_preds[k] for k in sorted(route_preds)], concat_dim=-1, name='concat_all')
            target = tf.concat([batch_route_tgt[k] for k in sorted(route_preds)], axis=-1)
            loss = get_loss(target, net.outputs, batch_weights)

            init = tf.global_variables_initializer()
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            save_path = 'models/{}_{}_final.npz'.format('train', interval)
            with tf.Session() as sess:
                sess.run(init)
                coord = tf.train.Coordinator()
                tl.files.load_and_assign_npz(sess, save_path, net)
                threads = tf.train.start_queue_runners(sess, coord)
                # all_ftrs, all_tms = sess.run([{k: v.outputs for k, v in ftrs.items()}, batch_times])
                # get_ftrs('train_ftrs', all_ftrs, all_tms)
                v_loss,y_true, pred = sess.run([loss,target, net.outputs])
                all_preds.append(pred)
                all_losses.append(v_loss)
                print('interval {}: val_loss {:.4f}'.format(interval, v_loss))
                coord.request_stop()
                coord.join(threads)
            sess.close()
            tf.reset_default_graph()
            tl.layers.clear_layers_name()
        pred=reduce(lambda x,y:x+y, all_preds, 0.0)/len(intervals)
        loss=mape_loss(y_true,pred)
        print('all loss', all_losses)
        print('merge loss', np.mean(loss))

    else:
        for test_interval in test_intervals:
            resname = 'res0531_{}.csv'.format(test_interval)
            batch_link_ftr, batch_link_id, batch_link_g, \
            batch_route_ftr, batch_route_id, batch_route_g, \
            batch_wea_ftr, batch_time_ftr,\
            batch_times,_,_ = get_inputs(data_dir, data_set, test_interval, batch_size=14, shuffle=False)
            inputs=batch_link_ftr, batch_link_id, batch_link_g, \
                    batch_route_ftr, batch_route_id, batch_route_g, \
                    batch_wea_ftr, batch_time_ftr
            route_preds,ftrs = inference(inputs, data, False, False)

            net = tl.layers.ConcatLayer([route_preds[k] for k in route_preds], concat_dim=-1, name='concat_all')
            route_preds = {k: v.outputs for k, v in route_preds.items()}
            route_preds['time'] = batch_times
            init = tf.global_variables_initializer()
            dp_dict = tl.utils.dict_to_one(net.all_drop)
            with tf.Session() as sess:
                sess.run(init)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                save_path = '../../model/{}_{}_final.npz'.format('train', test_interval)
                tl.files.load_and_assign_npz(sess, save_path, net)
                preds = sess.run(route_preds, feed_dict=dp_dict)
                get_submission(preds, resname)
                # all_ftrs, all_tms = sess.run([{k: v.outputs for k, v in ftrs.items()}, batch_times])
                # get_ftrs('test_ftrs', all_ftrs, all_tms)
                coord.request_stop()
                coord.join(threads)
            sess.close()
            tf.reset_default_graph()
            tl.layers.clear_layers_name()

        # merge
        p15 = pd.read_csv('../../result/res0531_15.csv')
        p20 = pd.read_csv('../../result/res0531_20.csv')
        p24 = pd.read_csv('../../result/res0531_24.csv')
        avgt = (
                   p15['avg_travel_time'] +
                   p20['avg_travel_time'] +
                   p24['avg_travel_time']
               ) / 3
        p15['avg_travel_time'] = avgt
        p15.to_csv('../../result/result_0531.csv', float_format='%.2f', index=None)