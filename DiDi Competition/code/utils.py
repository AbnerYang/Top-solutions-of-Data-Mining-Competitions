from __future__ import unicode_literals
import codecs
import re
import datetime
import pandas as pd
import numpy as np

from collections import defaultdict

prefix = '/home/lllcho/Documents/season_1/'

train_prefix = prefix + 'training_data/'
test_prefix = prefix + 'test_set_1/'


def get_cluster_map(cluster_map_file):
    """
    Get cluster map
    :param cluster_map_file: file name
    :return: cluster2id, id2cluster
    """
    with codecs.open(cluster_map_file) as f:
        lines = f.readlines()
        map_ = [line.strip().split('\t') for line in lines]
        cluster2id = dict([(cluster, int(id)) for cluster, id in map_])
        id2cluster = dict([(int(id), cluster) for cluster, id in map_])
    return cluster2id, id2cluster


def get_pios(file_path):
    with codecs.open(file_path) as f:
        first_level = set([])
        sencond_level = set([])
        pois = {}
        for line in f.readlines():
            poi = defaultdict(list)
            pass
            line = line.strip()
            cluster = line.split('\t')[0]
            p = re.compile(r'\t[0-9]+#')
            t = [(a[1:-1], b.split('\t')) for a, b in zip(p.findall(line), p.split(line)[1:])]
            for first_, second_ in t:
                poi[int(first_)].extend(second_)
            pois[cluster] = poi
            poi_all = defaultdict(list)
        for cluster, poi in pois.items():
            for first_, sencond_count in poi.items():
                poi_all[first_].extend(sencond_count)
    pass


def time2index(dtime):
    time_index = dtime.time().hour * 6 + int(dtime.time().minute / 10)
    return time_index


def get_orders(file_path):
    """
    Get order info for the given file in pandas format
    :param file_path: file name
    :return: orders dataframe with filed 'order_id','driver_id','pas_id','start_id',
    'dest_id','price','time' and 'time_idx'.
    """
    names = ['order_id', 'driver_id', 'pas_id', 'start_id', 'dest_id', 'price', 'time']
    orders = pd.read_table(file_path, '\t', header=None, names=names)
    for str_ in names[:5]:
        orders[str_] = orders[str_].map(lambda x: str(x))
    orders['time'] = orders['time'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    orders['time_idx'] = orders['time'].map(time2index)
    return orders


def get_weather(file_path):
    names = ['time', 'weather', 'temp', 'pm25']
    weathers = pd.read_table(file_path, '\t', header=None, names=names)
    weathers['time'] = weathers['time'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    weathers['time_idx'] = weathers['time'].map(time2index)
    return weathers


def get_traffic(file_path):
    names = ['district', 'tj1', 'tj2', 'tj3', 'tj4', 'time']
    traffic = pd.read_table(file_path, '\t', header=None, names=names)
    traffic['time'] = traffic['time'].map(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S"))
    traffic['time_idx'] = traffic['time'].map(time2index)
    for name in names[1:-1]:
        traffic[name] = traffic[name].map(lambda x: str(x).strip().split(':')[1])
    return traffic
