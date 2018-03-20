#coding:utf8
import numpy as np
import pandas as pd
import os
import feature
import datetime
import tools

sel_win = '10'

link_path = '../../data/dataSets/training/links (table 3).csv' #输入文件路径

def trasfer_y(Y):
    Z = []
    for y in Y:
        ps = y[1:-1].replace(' ','').split(',')
        t = []

        for pi in ps:
            t.append(float(pi))
        Z.append(np.array(t))
    return np.array(Z)

def get_route_length():
    df = pd.read_csv(link_path, sep=',')
    r_len = {}
    for line, row in df.iterrows():
        r_len[row.link_id - 100] = int(row.length)
    
    #print r_len
    return r_len

    
def get_Ta(ids):
    rout_dic = {'A2':1, 'A3':2, 'B1':3, 'B3':4, 'C1':5, 'C3':6}
    #rout_len = get_route_length()
    
    X = ids.values
    T = []
    for x in X:
        ps = x[0].split('#')
        ps1 = ps[2].split(' ')
        
        dst = datetime.datetime.strptime(ps[2], '%Y-%m-%d %H:%M:%S')
        week = dst.weekday()
    
        miu = int(ps1[1][0:2])*60 + int(ps1[1][3:5])
        miu = int(miu/10) #按10分钟来划分时间片
        key = ps[0]+str(ps[1])
        k = rout_dic[key]
        #t1 = int(24*60/feature.win)*(k-1) + miu
        #t1 = miu
        #rlen = rout_len[key]
        month = dst.month
        minute = dst.minute
        T.append([k, miu, week, month]) #方向，时间片, 星期几, 日期
    return T
    
def get_train_data():

    dx = pd.read_csv(feature.train_out_x_path, sep='\t')
    dy = pd.read_csv(feature.train_out_y_path, sep='\t')
    
    #dy = dy[ (dy['time'] >= '07:00:00') & (dy['time'] <= '18:00:00')] #过滤掉两端的时间数据

    day_filter = ['2016-09-30', '2016-10-01']
    dy = dy[ (dy['time'] >= '05:00:00') & (dy['time'] <= '21:00:00')] #过滤掉两端的时间数据
    dy = dy[~dy['date'].isin(day_filter)]
    
    dy_train = dy[dy['date'] < '2016-10-18']

    
    #dy_train = dy[ ( (dy['date'] < '2016-10-11') & (dy['date'] >= '2016-09-01') ) | (dy['date'] < '2016-08-22') ] 
     
               
    dy_test = dy[dy['date'] >= '2016-10-18']
    dy_test = dy_test[ ( (dy_test['time'] >= '07:30:00') & (dy_test['time'] <= '08:30:00')) | ( (dy_test['time'] >= '16:30:00') & (dy_test['time'] <= '17:30:00'))]
    
    #dy_test = dy_test[ ( (dy_test['time'] == '08:00:00')) | ( (dy_test['time'] == '17:00:00'))]
    
    dy_train = dy_train.drop(['date', 'time'], axis=1) 
    dy_test = dy_test.drop(['date', 'time'], axis=1)
    
    d_train = pd.merge(dy_train, dx, how = "left", on=['id'])
    d_test = pd.merge(dy_test, dx, how = "left", on=['id'])
    
    d_train.dropna(axis=0,how='any', inplace=True) #删除包含nan的行
    d_test.dropna(axis=0,how='any', inplace=True)
    
    #d_train = d_train.fillna(0.0)
    #d_test = d_test.fillna(0.0)
    
    
    train_ids = d_train[['id']]
    test_ids = d_test[['id']]
    
    
    train_x2 = get_Ta(train_ids)
    test_x2 = get_Ta(test_ids)
    
    y_train = trasfer_y(d_train['avg_travel_time'].values)
    y_test = trasfer_y(d_test['avg_travel_time'].values)
    
    
    d_train.drop(['id', 'avg_travel_time'], axis=1, inplace=True)
    d_test.drop(['id', 'avg_travel_time'], axis=1, inplace=True)
    

    #d_train, remove = tools.remove_cons_col(d_train)
    #d_test = d_test.drop(remove, axis=1)
    
    train_x1 = d_train.values
    test_x1 = d_test.values
    
    #---无效果
    #train_x1 = tran_X1(train_x1)
    #test_x1 = tran_X1(test_x1)

    pt = get_x1_parti(train_x1)
    X_train, y_train, w = merge_feature(train_x1, train_x2, y_train, pt, 'train', train_ids.values)
    X_test, y_test = merge_feature(test_x1, test_x2, y_test, pt, 'test')
    
    print 'all size:', dy.shape[0]
    print 'train size:', len(y_train)
    print 'test size:', len(y_test)
    print 'train nan:', np.isnan(X_train).sum()
    print 'test nan:',np.isnan(X_test).sum()
    print 'train feat size:', len(X_train[0])
    print 'test feat size:', len(X_test[0])
    
    
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)  
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)    
    
    
    return X_train, y_train, X_test, y_test, w
    #return X_test, y_test, X_train, y_train, w

#获取特征x1的分位点
def get_x1_parti(x1):
    t = []
    for i in xrange(0, len(x1[0])):
        t.append([])
        
    for x in x1:
        for i in xrange(0, len(x)):
            if x[i] > 0.0001:
                t[i].append(x[i])
    
    #par = [0.25, 0.5, 0.75]
    par = []
    for i in xrange(0, 100):
        pi = i*0.1
        if pi > 0.999:
            break
        par.append(pi)
    
        
    e = []
    for i in xrange(0, len(x1[0])):
        t[i].sort()
        ei = []
        for pi in par:
            ei.append(t[i][int(pi*len(t[i]))])
        e.append(ei)
    #print 'partition:',e
    return e
   
#sli = 1
def count_zero(x):
    i = 0
    mean = []
    for xi in x:
        if xi < 0.0001:
            i += 1
        else:
            mean.append(xi)
    return i, np.array(mean).mean(), np.array(mean).std(), np.array(mean).min()



def cal_y_mean_std(y, x2):
    yt = []
    for i in range(6):
        yt.append([])
        
    for i in range(len(y)):
        ya = y[i]
        for yi in ya:
            if yi > 0.0001:
                yt[x2[i][0]-1].append(yi)
    
    mean = []
    std = []
    
    for i in xrange(6):
        t = np.array(yt[i])
        mean.append(t.mean())
        std.append(t.std())
    return mean, std       
        

def merge_feature(x1, x2, y, pt, flag = 'train', ids=None):
    
    
    X = []
    Y = []
    
    wei = []
    #dic = read_tra_sli_vec()

    
    #feat_y = read_feat_y()
    #mape = get_mape()
    mi = [0]*6
    cti = [0]*6

    mean, std = cal_y_mean_std(y, x2)

    #rou_len = get_route_length()
    
    for i in xrange(0, len(y)):
        
        cnt, mean1, std1, min1 = count_zero(y[i])
        
        #my = []
        for j in xrange(0, len(y[0])):
            if y[i][j] > 0.0001:

                if flag == 'train':
    
                    #wa = [7.0, 8.0, 7.0, 100.0, 100.0, 100.0]
                    #wb = [3, 100, 100.0, 100, 100, 100]
                    wa = [6.0]*6
                    wb = [5.0]*6
                    fan = x2[i][0] - 1
                    
                    if y[i][j] > mean[fan] + wa[fan]*std[fan]:
                        #y[i][j] = mean[fan] + wa[fan]*std[fan]
                        mi[fan] += 1
                        continue
                    if y[i][j] < mean[fan] - wb[fan]*std[fan]:
                        cti[fan] += 1
                        continue
                
                
                #x = list(x1[i])[24:] + [j]
                x = list(x1[i]) + [j]                
                
                x.append(x2[i][0]) #方向
                
                
                x.append(x2[i][1]+j*2) #时间片123
                hour = int((x2[i][1]+j*2)/6)
                x.append(hour) #小时124
                x.append(int((x2[i][1]+j*2)/3)) #半小时     125   
                        
                x.append(x2[i][2]) #星期几
    
                #是否周末
                if x2[i][2] == 5 or x2[i][2] == 6:
                    x.append(1)
                else:
                    x.append(0)
                
                
                x.append(x2[i][3]) #月份
                        
                X.append(x)
                
                Y.append(y[i][j])
                  
                #Y.append(y[i][j]/mean)

    print 'mi:', mi
    print 'cti:', cti
#    print flag + ' mean result:', np.array(loss).mean()
    if flag == 'train':
        return X, Y, wei
    return X, Y


def get_train_val():
    dx = pd.read_csv(feature.train_out_x_path, sep='\t')
    dy = pd.read_csv(feature.train_out_y_path, sep='\t')
    
    d_test = pd.read_csv(feature.test_out_x_path, sep='\t')
    
    #dy_train = dy[ (dy['time'] >= '05:00:00') & (dy['time'] <= '21:00:00')] #过滤掉两端的时间数据
    
    day_filter = ['2016-09-30', '2016-10-01']
    dy = dy[ (dy['time'] >= '05:00:00') & (dy['time'] <= '21:00:00')] #过滤掉两端的时间数据
    dy_train = dy[~dy['date'].isin(day_filter)]
                 
    dy_train = dy_train.drop(['date', 'time'], axis=1) 
    d_train = pd.merge(dy_train, dx, how = "left", on=['id'])
    
    
    
    d_train.dropna(axis=0, how='any', inplace=True) #删除包含nan的行
    d_test.dropna(axis=0, how='any', inplace=True)
    
    #d_train = d_train.fillna(0.0)
    #d_test = d_test.fillna(0.0)
    
    
    train_ids = d_train[['id']]
    test_ids = d_test[['id']]
    
    
    train_x2 = get_Ta(train_ids)
    test_x2 = get_Ta(test_ids)
    
    y_train = trasfer_y(d_train['avg_travel_time'].values)
    
    
    d_train.drop(['id', 'avg_travel_time'], axis=1, inplace=True)
    d_test.drop(['id'], axis=1, inplace=True)
    

    #d_train, remove = tools.remove_cons_col(d_train)
    #d_test = d_test.drop(remove, axis=1)

    train_x1 = d_train.values
    test_x1 = d_test.values

    X_train, y_train, w = merge_feature(train_x1, train_x2, y_train, 'train')
    X_test, id_test = merge_feature_val(test_x1, test_x2, test_ids)
    
    print 'all size:', dy.shape[0]
    print 'train size:', len(y_train)
    print 'test size:', len(id_test)
    print 'train nan:', np.isnan(X_train).sum()
    print 'test nan:',np.isnan(X_test).sum()
    print 'feat size:', len(X_train[0])
    
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)  
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, y_train, X_test, id_test

    
def merge_feature_val(x1, x2, ids):
    X = []
    L = []
    ids = ids.values
    
    #dic = read_tra_sli_vec()

    for i in xrange(0, len(x1)):
        
        
        ps1 = ids[i][0].split('#')
        ps2 = ps1[2].split(' ')

        if ps2[1] == '08:00:00' or ps2[1] == '17:00:00':

                  
            dt = datetime.datetime.strptime(ps1[2], '%Y-%m-%d %H:%M:%S')
            for j in xrange(0, 6):

                #x = list(x1[i])[24:] + [j]
                x = list(x1[i]) + [j]                
                
                x.append(x2[i][0]) #方向
                
                
                x.append(x2[i][1]+j*2) #时间片123
                hour = int((x2[i][1]+j*2)/6)
                x.append(hour) #小时124
                x.append(int((x2[i][1]+j*2)/3)) #半小时     125   
                        
                x.append(x2[i][2]) #星期几

                #是否周末
                if x2[i][2] == 5 or x2[i][2] == 6:
                    x.append(1)
                else:
                    x.append(0)
                
                
                x.append(x2[i][3]) #月份
                
                X.append(x)
                
                t = [ps1[0], ps1[1]]
                dw1 = datetime.timedelta(minutes=20*j)
                stime1 = (dt+dw1).strftime('%Y-%m-%d %H:%M:%S')
                dw2 = datetime.timedelta(minutes=20*(j+1))
                stime2 = (dt+dw2).strftime('%Y-%m-%d %H:%M:%S')
                t.append('['+ stime1 + ',' + stime2 + ')')
                L.append(t)
    return X, L


def get_mape():
    df = pd.read_csv('data/utrain.pred.y.mape.tsv', sep='\t', header=None)
    X = df.values
    t = []
    for x in X:
        t.append([x[1], x[2]])
    return t
    

def read_weather():
    df = pd.read_csv('data/weather.tsv', sep=',')
    X = df.values
    
    dic = {}
    day = ''
    hour = 0
    for x in X:
        x = list(x)
        if day != '' and x[1] - hour == 3 and x[1] == day:
            dic[day+'#'+str(hour+2)] = x[2:]
        
        dic[x[0]+'#'+str(x[1])] = x[2:]
        dic[x[0]+'#'+str(x[1]+1)] = x[2:]
        
        day = x[0]
        hour = int(x[1])
        
    return dic

if __name__ == '__main__':
    read_weather()
    #get_route_length()
    #get_mape()
    
    