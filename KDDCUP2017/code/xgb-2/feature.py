#coding:utf-8
import pandas as pd
import numpy as np
import os
import os.path

import datetime

win = 10 #滑动窗口
N1 = 120/win
interval = 20 #2小时切分间隔
N2 = 120/interval

#data_dir = 'data/dataSets'


route_in_path = '../../data/dataSets/training/routes (table 4).csv' #输入文件路径
train_in_path = '../../feature/xgb-2/train.csv' #输入文件路径
test_in_path = '../../feature/xgb-2/test.csv' #输入文件路径

train_out_x_path = '../../feature/xgb-2/train.x.tsv' #输出文件路径
train_out_y_path = '../../feature/xgb-2/train.y.tsv' #输出文件路径

test_out_x_path = '../../feature/xgb-2/test.x.tsv' #输出文件路径

def start():
    train = pd.read_csv('../../data/dataSets/training/trajectories(table 5)_training.csv', header = 0)
    test_1 = pd.read_csv('../../data/dataSets/test_1/trajectories(table_5)_training2.csv', header = 0)
    test_2 = pd.read_csv('../../data/dataSets/test_2/trajectories(table 5)_test2.csv', header = 0)
    train = train.append(test_1)
    train.to_csv(train_in_path, index = False)
    test_2.to_csv(test_in_path, index = False)




def get_datetime_a(start_time, nmin):
    dst = datetime.datetime.strptime(start_time[0:14]+ '00:00', '%Y-%m-%d %H:%M:%S')
    return dst + datetime.timedelta(minutes=nmin)

def extract_target():
    dy = {}
    dn = {}

    df = pd.read_csv(train_in_path, sep=',')
    for i, row in df.iterrows():
        start_time = row.starting_time
        dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        nwin = int(start_time[-5:-3])/win
        #dst = datetime.datetime.strptime(start_time[0:14]+ str(nwin*win) + ':00', '%Y-%m-%d %H:%M:%S')
        dst = get_datetime_a(start_time, nwin*win)
        
        for j in xrange(0, N1):
            dtime = (dst - datetime.timedelta(minutes=win*j))
            stime = dtime.strftime('%Y-%m-%d %H:%M:%S')
            key = row.intersection_id + '#' + str(row.tollgate_id) + '#' + stime
            
            t = int((dt - dtime).seconds/60/20)
            
            if key not in dy:
                dy[key] = [0.0]*6
                dn[key] = [0]*6
            
            dy[key][t] += row.travel_time
            dn[key][t] += 1
                
            #print stime, t
        #break
        if i%1000 == 0:
            print i
    X = []
    for k, v in dy.items():
        ps1 = k.split('#')
        ps2 = ps1[2].split(' ')
        x = [k, ps2[0], ps2[1]]
        xi = dy[k]
        for i in xrange(0, 6):
            if dn[k][i] > 0:
                xi[i] = dy[k][i]/dn[k][i]
        x.append(xi)
        X.append(x)
        
    df = pd.DataFrame(X, columns=['id', 'date', 'time', 'avg_travel_time'])
    df.to_csv(train_out_y_path,sep='\t', index=False)


#读取路径
def read_route():
    df = pd.read_csv(route_in_path, sep=',')
    route_dic = {}
    for i, row in df.iterrows():
        key = row.intersection_id + '#' + str(row.tollgate_id)
        x = [0]*24
        ps = row.link_seq.split(',')
        for pi in ps:
            x[int(pi) - 100] = 1
        route_dic[key] = x
    return route_dic 

'''
#保存汽车序列信息
def save_tra_cnt_seq(dn, p):
    route_dic = read_route()
    X = []
    for k, v in dn.items():
        x = [k]
        ps = k.split('#')
        key = ps[0]+'#'+ps[1]
        for i in xrange(0, N2):
            x += list(np.array(route_dic[key])*v[i])
        X.append(x)
        
    titles = ['id']
    for i in xrange(0, N2):
        for j in xrange(0, 24):
            titles.append('tra_traCnt'+str(i)+'_'+str(j))
    df = pd.DataFrame(X, columns=titles)
    ypath = os.path.join(data_dir, flags[p] +'.tracnt.tsv')
    df.to_csv(ypath,sep='\t', index=False)  #保存每个路段interval内的通过车辆数
'''
    

def extract_traj(p):
    dx = {}
    dn = {}
    
    #dxmin = {}
    #dxmax = {}
    dall = {}
    dall_n = {}
    
    dmd = {}

    se = set()    
    
    filepath = train_in_path
    if p == 1:
        filepath = test_in_path
    
    df = pd.read_csv(filepath, sep=',')
    for i, row in df.iterrows():
        start_time = row.starting_time
        dt = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        nwin = int(start_time[-5:-3])/win + 1
        dst = get_datetime_a(start_time, nwin*win)
        
        p1s = row.travel_seq.split(';')
        ts = {}
        for pi in p1s:
            p2s = pi.split('#')
            ts[int(p2s[0][1:])] = float(p2s[2])
        
        for j in xrange(0, N1):
            dtime = (dst + datetime.timedelta(minutes=win*j))
            stime = dtime.strftime('%Y-%m-%d %H:%M:%S')
            key = row.intersection_id + '#' + str(row.tollgate_id) + '#' + stime
            se.add(row.intersection_id + '#' + str(row.tollgate_id) + '#' + stime.split(' ')[0]) #路径+日期
            t = int((dtime - dt).seconds/60/interval)

            
            if key not in dx:
                dx[key] = [0.0]*24*N2
                dn[key] = [0]*N2
                dall[stime] = [0.0]*24*N2
                dall_n[stime] = [0]*N2
                      
                dmd[key] = []
                for si in xrange(0, 24*N2):
                    dmd[key].append([])
                #dxmin[key] = [9999]*24*N2
                #dxmax[key] = [0.0]*24*N2
           
            
            dall_n[stime][N2-1-t] += 1
            
            dn[key][N2-1-t] += 1
            for k, v in ts.items():
                dx[key][(N2-1-t)*24 + k] += v
                  
                dall[stime][(N2-1-t)*24 + k] += v
                    
                dmd[key][(N2-1-t)*24 + k].append(v)
                #dxmin[key][(N2-1-t)*24 + k] = min(dxmin[key][(N2-1-t)*24 + k], v)
                #dxmax[key][(N2-1-t)*24 + k] = max(dxmax[key][(N2-1-t)*24 + k], v)
            
        #break
        if i%1000 == 0:
            print i
            #break

    X = []
    
    for k, v in dall.items():
        for i in xrange(0, len(v)):
            if dall_n[k][int(i/24)] > 0 :
                v[i] = v[i]/dall_n[k][int(i/24)]
        dall[k] = v
    
    for k, v in dx.items():   
        '''
        #中位数
        v1 = [0.0]*len(v)
        for i in xrange(0, len(v)):
            if len(dmd[k][i]) > 0:
                v1[i] = np.array(dmd[k][i]).std()
        '''
        
        for i in xrange(0, len(v)):
            if dn[k][int(i/24)] > 0:
                v[i] = v[i]/dn[k][int(i/24)]
        
        x = [k] + v # + dall[ps[2]] #+ dn[k]  + dall[k]
        X.append(x)
    
    
    #---补充没有车辆通过的训练数据
    ock = [' 08:00:00', ' 17:00:00']
    for si in se:
        for oi in ock:
            key = si+oi
            if key not in dx:
                print key
                X.append([key] + [0.0]*(len(X[0])-1) )
    
    
    #补充无车辆通过时间段
    X = add_missing(X)
    
    
    titles = ['id']
    
    for i in xrange(0, len(X[0])-1):
        titles.append('tra_time_'+str(i))
    '''
    for i in xrange(0, N2):
        for j in xrange(0, 24):
            titles.append('tra_time'+str(i)+'_'+str(j)+'a')
            titles.append('tra_time'+str(i)+'_'+str(j)+'b')
    '''
    df = pd.DataFrame(X, columns=titles)
    return df
        
    #save_tra_cnt_seq(dn, p)


def add_missing(X):
    route_dic = read_route()
    T = []
    for x in X:
        ps = x[0].split('#')
        key = ps[0]+'#'+ps[1]
        t = [x[0]]
        z = route_dic[key]
        for i in xrange(0, N2):
            for j in xrange(0, 24):
                #t.append(x[1+24*i+j])
                
                if  z[j] > 0 and x[1+24*i+j] < 0.0001:
                    #t.append(1)
                    t.append(-1)
                else:
                    #t.append(0)
                    t.append(x[1+24*i+j])
        t += x[24*N2:]
        T.append(t)
    return T


def extract_x(flag=0):
    global interval
    global N2
    
    iers = [20]
    dfs = []
    for ie in iers:
        interval = ie
        N2 = 120/interval
        dfi = extract_traj(flag)
        dfs.append(dfi)
    df = dfs[0]
    for i in xrange(1, len(dfs)):
        df = pd.merge(df, dfs[i], how='left', on=['id'])
        
    outpath = train_out_x_path
    if flag == 1:
        outpath = test_out_x_path

    df.to_csv(outpath, sep='\t', index=False)  #保存每个路段interval内的平均时间

if __name__ == "__main__":
    start()
    extract_target()
    
    extract_x(0)
    extract_x(1)
    #extract_traj('trajectories(table 5)_training.csv', 0)
    #extract_traj('trajectories(table 5)_test1.csv', 1)