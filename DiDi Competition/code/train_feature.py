from __future__ import unicode_literals
import h5py
import numpy as np
import utils

prefix = '/home/lllcho/Documents/season_1/'
train_prefix = prefix + 'training_data/'
test_prefix = prefix + 'test_set_1/'
cluster2id, id2cluster = utils.get_cluster_map(test_prefix + 'cluster_map/cluster_map')

dates,time_idx,dests=21,288,66
train_orders_dest = np.zeros((dates, time_idx, dests, 67), dtype=np.float32)  # 去向地
train_orders_price = np.zeros((dates, time_idx, dests, 67), dtype=np.float32)  # 价格
train_orders_gap = np.zeros((dates, time_idx, dests), dtype=np.float32)  # gap
train_weathers = np.zeros((dates, time_idx, 3), dtype=np.float32)
train_traffics = np.zeros((dates, time_idx, dests, 4), dtype=np.float32)

for idx, date in enumerate(range(1, 22)):
    print ('date {0:0>2}'.format(date))
    order = utils.get_orders(train_prefix + 'order_data/order_data_2016-01-{0:0>2}'.format(date))
    order = order.loc[:, ['driver_id', 'start_id', 'dest_id', 'price', 'time_idx']]
    order['start_id'] = order['start_id'].map(lambda x: cluster2id[x])
    order['dest_id'] = order['dest_id'].map(lambda x: cluster2id[x] if x in cluster2id.keys() else 0)

    dest = np.zeros((time_idx, dests, 67), dtype=np.float32)
    price = np.zeros((time_idx, dests, 67), dtype=np.float32)
    gap = np.price = np.zeros((time_idx, dests), dtype=np.float32)
    for i in range(order.shape[0]):
        t = order.iloc[i]
        if t['driver_id'] == 'nan':
            gap[t['time_idx'], t['start_id'] - 1] += 1
        else:
            dest[t['time_idx'], t['start_id'] - 1, t['dest_id']] += 1
            price[t['time_idx'], t['start_id'] - 1, t['dest_id']] += t['price']

    train_orders_dest[idx] = dest
    train_orders_price[idx] = price
    train_orders_gap[idx] = gap

    weather = utils.get_weather(train_prefix + 'weather_data/weather_data_2016-01-{0:0>2}'.format(date))
    weather = weather.iloc[:, 1:].groupby('time_idx').apply(np.mean)
    weather = weather.reindex(range(time_idx), method='nearest')
    weather = weather.as_matrix(['weather', 'temp', 'pm25'])
    train_weathers[date - 1] = weather.astype(np.float32)

    traffic = utils.get_traffic(train_prefix + 'traffic_data/traffic_data_2016-01-{0:0>2}'.format(date))
    traffic['district'] = traffic['district'].map(lambda x: cluster2id[x])
    group = traffic.groupby('district')
    dists = traffic.district.unique().tolist()
    traffs = np.zeros((time_idx, dests, 4), dtype=np.float32)
    for dist in dists:
        traff = group.get_group(dist)
        traff = traff.loc[:, ['tj1', 'tj2', 'tj3', 'tj4', 'time_idx']]
        traff = traff.groupby('time_idx').apply(np.mean).reindex(range(time_idx), method='nearest')
        traff = traff.as_matrix(columns=['tj1', 'tj2', 'tj3', 'tj4'])
        traffs[:, dist - 1, :] = traff.astype(np.float32)
    train_traffics[idx] = traffs

with h5py.File('train_data.h5', 'w') as f:
    f.create_dataset('dest', data=train_orders_dest)
    f.create_dataset('price', data=train_orders_price)
    f.create_dataset('gap', data=train_orders_gap)
    f.create_dataset('weathers', data=train_weathers)
    f.create_dataset('traffics', data=train_traffics)
