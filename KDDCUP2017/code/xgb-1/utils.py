# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from copy import deepcopy,copy
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

# Add time information weekday/date/hour/minute
def translateTime(df,name):
    minuteList = []
    hourList = []
    dateList = []
    weekList = []
    time = df[name].values
    for t in time:
        date = datetime.strptime(t,'%Y-%m-%d %H:%M:%S')
        dateList.append(int(str(date.month).zfill(2)+str(date.day).zfill(2)))
        weekList.append(date.weekday())
        minuteList.append(date.minute)
        hourList.append(date.hour)
    df['weekday'] = weekList
    df['date'] = dateList
    df['hour'] = hourList
    df['minute'] = minuteList
    return df
    
def getLinkSpeedFeature(df, link_dic):
    mydf = deepcopy(df)
    mean = mydf.describe().travel_time['mean']
    std = mydf.describe().travel_time['std']

    mydf['mean'] = mean
    mydf['std'] = std

    filtered = mydf[abs(mydf['travel_time'] - mydf['mean']) <= 5 * mydf['std']].drop(['mean', 'std'], axis=1)

    new = filtered.groupby(['date', 'hour', 'intersection_id', 'tollgate_id', 'link']).mean().reset_index()
    new = new.pivot_table(index=['date', 'hour', 'intersection_id', 'tollgate_id'], columns='link',
                          values='travel_time') \
        .reset_index().fillna(0)

    colName = ['linkTime_' + str(i) for i in range(100, 124)]
    colName = ['date', 'hour', 'intersection_id', 'tollgate_id'] + colName
    new.columns = colName

    for l in link_dic.keys():
        new['linkSpeed_' + str(l)] = new['linkTime_' + str(l)] / link_dic[l]['length']
    new = new.drop(['linkTime_' + str(i) for i in range(100, 124)], axis=1)

    new2 = deepcopy(new)
    new2['hour'] = new2['hour'] - 1
    new2 = pd.merge(new, new2, on=['date', 'hour', 'intersection_id', 'tollgate_id'], how='outer').fillna(0)
    new2['hour'] = new2['hour'] + 2

    return new2


# Outlier filter
def outlierFilter(df):
    df = df.drop_duplicates()
    df2 = deepcopy(df)
    df2['window'] = df2.minute // 20 * 20
    std = df2.describe().travel_time['std']
    mean = df2.describe().travel_time['mean']
    df2 = df2[abs(df2.travel_time - mean) <= 5 * std]
    std = df2['travel_time'].groupby([df2.date, df2.hour, df2.intersection_id, df2.tollgate_id, df2.window]) \
        .agg([np.std, np.mean]).reset_index().fillna(0)
    check = pd.merge(df2, std, on=['date', 'hour', 'intersection_id', 'window', 'tollgate_id'], how='left')
    filtered = check[abs(check['mean'] - check.travel_time) <= (5 * check['std'])]
    filtered.drop(['std', 'mean', 'window'], axis=1, inplace=True)
    return filtered


# Calculate the average travel time according to time stamp
def aggregateTravelTime(df, timeStamp):
    df['time_window'] = df['minute'] // timeStamp * timeStamp
    mydf = df['travel_time'].groupby(
        [df.date, df.hour, df.time_window, df.tollgate_id, df.intersection_id]).mean().reset_index()
    return mydf


# Set up time_window
def getTimeWindows(df):
    n = df.shape[0]
    date = df.date.values
    hour = df.hour.values
    window = df.time_window.values
    windowList = []
    for i in range(n):
        windowList.append(str(hour[i]).zfill(2) + str(window[i]).zfill(2))
    df['time_window'] = windowList
    return df


# Transeform df to time series
def transformLongTimeSeries(df):
    ids = df[['tollgate_id', 'intersection_id']].drop_duplicates()
    k = ids.shape[0]
    mydf = pd.DataFrame()
    for i in range(k):
        inter = ids.intersection_id.values[i]
        toll = ids.tollgate_id.values[i]
        ts = df[(df.intersection_id == inter) & (df.tollgate_id == toll)]
        ts = ts.pivot(index='date', columns='time_window', values='travel_time').reset_index()
        ts['tollgate_id'] = toll
        ts['intersection_id'] = inter
        mydf = mydf.append(ts)
    return mydf


# Generate a 2hr time series
def generateShortTimeSeries(df, timeStamp, times, method = 'train'):
    timeSeries = pd.DataFrame()
    minute = [str(i).zfill(2) for i in range(0, 60, timeStamp)]
    for h in times:
        colName = []
        for m in minute:
            colName.append(str(h).zfill(2) + m)
        for m in minute:
            colName.append(str(h + 1).zfill(2) + m)
        colName = colName + ['date', 'tollgate_id', 'intersection_id']
        mydf = df[colName]
        if method == 'label':
            mydf.columns = [str(x) for x in range(0, 120, 20)] + ['date', 'tollgate_id', 'intersection_id']
        elif method == 'train':
            mydf.columns = ['stamp_' + str(timeStamp) + '_' + str(i) for i in range(0, 120, timeStamp)] + ['date', 'tollgate_id', 'intersection_id']
        mydf['hour'] = h
        timeSeries = timeSeries.append(mydf)
        timeSeries = timeSeries.reset_index()
        timeSeries.drop('index', axis=1, inplace=True)
    return timeSeries


# Fill nan with median value
def fillNan(df, method='median'):
    mydf = df.drop(['date', 'tollgate_id', 'intersection_id', 'hour'], axis=1)
    trans = mydf.T
    m = trans.shape[1]
    des = trans.describe().values
    if method == 'median':
        values = des[5]
    elif method == 'mean':
        values = des[1]
    else:
        print('Wrong method!')
        return 0
    for i in range(m):
        mydf.iloc[i, :].fillna(values[i], inplace=True)
    mydf = mydf.join(df[['date', 'tollgate_id', 'intersection_id', 'hour']])
    return mydf


# Get stat features
def getStatFeature(df, timeStamp):
    indexs = ['tollgate_id', 'intersection_id', 'hour', 'date']
    temp = df[indexs]
    df = df.drop(indexs, axis=1)
    mydf = deepcopy(df)
    mydf[str(timeStamp) + '_mean'] = df.mean(axis=1)
    mydf[str(timeStamp) + '_median'] = df.median(axis=1)
    mydf[str(timeStamp) + '_max'] = df.max(axis=1)
    mydf[str(timeStamp) + '_min'] = df.min(axis=1)
    mydf[str(timeStamp) + '_std'] = df.std(axis=1)
    mydf[str(timeStamp) + '_range'] = df.max(axis=1) - df.min(axis=1)
    mydf[str(timeStamp) + '_%25'] = df.quantile(0.25, axis=1)
    mydf[str(timeStamp) + '_%75'] = df.quantile(0.75, axis=1)
    return mydf.join(temp)


# Get date
def getDate(df):
    mydf = copy(df)
    date = df.date.values
    dateList = []
    for d in date:
        time = datetime.strptime(d, '%Y-%m-%d')
        dateList.append(int(str(time.month) + str(time.day).zfill(2)))
    mydf['date'] = dateList
    return mydf

# Get weather feature1
def getWeatherFeature(feature, weather):
    weathers = deepcopy(weather)
    weathers = getDate(weathers)
    feature['hour1'] = (feature['hour'] + 2) // 3 * 3
    feature['hour2'] = feature['hour1'] + 3
    weathers = weathers.rename(columns={'hour': 'hour1'})
    feature = pd.merge(feature, weathers, on=['date', 'hour1'], how='left')
    weathers = weathers.rename(columns={'hour1': 'hour2'})
    feature = pd.merge(feature, weathers, on=['date', 'hour2'], how='left')
    return feature


# Get common features
def getCommonFeature(df, indexName, timeStamps, setup='train'):
    flag = True
    for stamp in timeStamps:
        time = aggregateTravelTime(df, stamp)
        time = getTimeWindows(time)
        ts = transformLongTimeSeries(time)
        if setup == 'train':
            times = range(6, 20)
        elif setup == 'test':
            times = [6, 15]
        train = generateShortTimeSeries(ts, stamp, times)
        train = fillNan(train)
        train = getStatFeature(train, stamp)
        if flag == True:
            trainFeature = train
            flag = False
            continue
        else:
            trainFeature = pd.merge(trainFeature, train, on=indexName, how='left')

    le = LabelEncoder()
    trainFeature['intersection'] = le.fit_transform(trainFeature['intersection_id'])

    date = trainFeature.date.values
    n = trainFeature.shape[0]
    weekList = []
    for i in range(n):
        day = datetime.strptime('2016' + str(date[i]).zfill(2), '%Y%m%d')
        weekList.append(day.weekday())
    trainFeature['weekday'] = weekList

    return trainFeature


# Get train set
def getTrainSet(feature, label):
    feature['hour'] = feature['hour'] + 2
    df = pd.merge(label, feature, on=['date', 'tollgate_id', 'hour', 'intersection_id'], how='left')
    n = df.shape[0]
    y = [str(i) for i in range(0, 120, 20)]
    feature = df.drop(y, axis=1)
    label = df[y]
    trainFeatures = pd.DataFrame()

    for i in range(6):
        y_i = label[y[i]].values
        indexFeature = pd.DataFrame(np.zeros([n, 6]))
        indexFeature[i] = 1
        Features = feature.join(indexFeature)
        Features['label'] = y_i
        trainFeatures = trainFeatures.append(Features)

    return trainFeatures


# Get testSet
def getTestSet(df):
    n = df.shape[0]
    df['hour'] = df['hour'] + 2
    hour = df.hour.values
    date = df.date.values
    feature = df
    minute = [0, 20, 40]
    Features = pd.DataFrame()
    for i in range(6):
        indexFeature = pd.DataFrame(np.zeros([n, 6]))
        indexFeature[i] = 1
        timeList = []

        for j in range(n):
            d = int(date[j] % 100)
            h = int(hour[j]) + i // 3
            m = minute[i % 3]
            start_time = str(datetime(2016, 10, d, h, m, 0))
            end_time = str(datetime(2016, 10, d, h, m, 0) + timedelta(minutes=20))
            timeList.append('[' + start_time + ',' + end_time + ')')
        indexFeature['time_window'] = timeList
        Features = Features.append(feature.join(indexFeature))

    return Features


# Link seq filter
def LinkseqFilter(df,route):
    df['seq_count'] = df.travel_seq.apply(lambda x:len(x.split(';')) ).astype(int)
    route['count'] = route.link_seq.apply(lambda x:len(x.split(','))).astype(int)
    df = pd.merge(df,route.drop('link_seq',axis=1),on=['intersection_id','tollgate_id'],how='left')
    df = df[df.seq_count==df['count']].drop('count',axis=1)
    return df

# Get travel volume
def aggregateTravelVolume(df,timeStamp):
    df1 = deepcopy(df)
    df1['time_window'] = df1['minute']//timeStamp*timeStamp
    mydf = df1.groupby([df1.date,df.hour,df1.time_window,df1.tollgate_id,df1.intersection_id]).size().reset_index()
    mydf = mydf.rename(columns = {0:'volume'})
    return mydf

# Set a time_pointer
def getTime(df):
    mydf = deepcopy(df)
    mydf['time'] = mydf['hour'].apply(lambda x:str(x).zfill(2)) + mydf['time_window'].apply(lambda x:str(x).zfill(2))
    return mydf

# Get short time series for volume feature
def getVolumeTimeSeries(df,setup='train'):
    ids = ['date','tollgate_id','intersection_id']
    if setup=='train':
        timeList = range(6,19)
    elif setup=='test':
        timeList = [6,15]
    mydf = pd.DataFrame()
    for t in timeList:
        time = [str(t).zfill(2)+str(x).zfill(2) for x in [0,20,40]] +\
        [str(t+1).zfill(2)+str(x).zfill(2) for x in [0,20,40]]
        ts = df[ids+time]
        ts.columns = ids + range(0,120,20)
        ts['hour'] = t+2
        mydf = mydf.append(ts)
    return mydf

# Get volume feature
def getVolumeFeatures(df):
    df['60_volume1'] = df[0] + df[20] + df[40]
    df['60_volume2'] = df[60] + df[80] +df[100]
    df['40_volume1'] = df[0] + df[20]
    df['40_volume2'] = df[40] + df[60]
    df['40_volume3'] = df[80] + df[100]
    df['sum_volume'] = df[0] + df[20] + df[40] + df[60] + df[80] +df[100]
    return df

# Get link index
def getLink(route,links):
    data1 = links.values
    matrix = np.zeros([6,25])
    for i in range(6):
        sum_length = 0
        seqs = route.link_seq.values[i].split(',')
        for link in seqs:
            matrix[i,int(link)%100] = 1
            sum_length += data1[int(link)%100,1]
        matrix[i,24] = sum_length
    mydf = pd.DataFrame(matrix)
    colName = ['link_'+str(i) for i in range(100,124)]
    colName.append('sum_length')
    mydf.columns = colName
    mydf = mydf.join(route[['intersection_id','tollgate_id','count']])
    return mydf

# Get link feature
def getLinkFeature(link_dic, df):
    mydf = deepcopy(df)
    for link in link_dic.keys():
        mydf['link_' + str(link) + '_length'] = mydf['link_' + str(link)] * link_dic[link]['length']
        # mydf['link_'+str(link)+'_lanes'] = mydf['link_'+str(link)] * link_dic[link]['lanes']
        mydf['link_' + str(link) + '_area'] = mydf['link_' + str(link)] * link_dic[link]['area']
    return mydf


# Split travel_seq into links seq
def splitSeq(df):
    data = df[['intersection_id', 'tollgate_id', 'travel_seq']].values
    n = df.shape[0]
    lst = []
    for i in range(n):
        intersection = data[i][0]
        tollgate = data[i][1]
        seq = data[i][2].split(';')
        for link in seq:
            l = link.split('#')
            lst.append([intersection, tollgate, l[0], l[1], float(l[2])])
    mydf = pd.DataFrame(lst)
    mydf.columns = ['intersection_id', 'tollgate_id', 'link', 'time', 'travel_time']
    return mydf

# Split time from links seq
def splitTime(df):
    mydf = deepcopy(df)
    daylst = []
    hourlst = []
    minutelst = []
    time = mydf.time.values
    for t in time:
        t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        hourlst.append(t.hour)
        daylst.append(str(t.month).zfill(2) + str(t.day).zfill(2))
        minutelst.append(t.minute)
    mydf['minute'] = minutelst
    mydf['hour'] = hourlst
    mydf['date'] = daylst
    mydf['window'] = mydf['minute'] // 20 * 20
    return mydf

# Get
def getLinkVolumeFeature(mydf, link_dic):
    df1 = mydf.groupby(['date', 'intersection_id', 'tollgate_id', 'hour', 'link']).size().reset_index()
    volume = df1.pivot_table(index=['date', 'intersection_id', 'tollgate_id', 'hour'], columns='link', values=0) \
        .reset_index().fillna(0)
    # print volume.columns
    colName = ['linkVol_' + str(i) for i in range(100, 124)]
    colName = ['date', 'intersection_id', 'tollgate_id', 'hour'] + colName
    volume.columns = colName

    volume['sumVol'] = 0
    volume['sumLen'] = 0
    volume['sumArea'] = 0
    for l in link_dic.keys():
        volume['linkVol_' + str(l) + '_len'] = volume['linkVol_' + str(l)] / link_dic[l]['length']
        # volume['linkVol_'+str(l)+'_lane'] = volume['linkVol_'+str(l)] /link_dic[l]['lanes']
        volume['linkVol_' + str(l) + '_area'] = volume['linkVol_' + str(l)] / link_dic[l]['area']

        volume['sumLen'] += volume['linkVol_' + str(l)] / volume['linkVol_' + str(l)] * link_dic[l]['length']
        volume['sumArea'] += volume['linkVol_' + str(l)] / volume['linkVol_' + str(l)] * link_dic[l]['area']
        volume['sumVol'] += volume['linkVol_' + str(l)]

    volume['per_len'] = volume['sumVol'] / volume['sumLen']
    volume['per_area'] = volume['sumVol'] / volume['sumArea']

    volume2 = deepcopy(volume)
    volume2['hour'] = volume2['hour'] - 1
    volume3 = pd.merge(volume, volume2, on=['date', 'intersection_id', 'tollgate_id', 'hour'], how='outer')
    volume3['hour'] = volume3['hour'] + 2

    volume3.drop(['sumLen_y', 'sumArea_y'], axis=1, inplace=True)
    volume3['2per_len'] = volume3['sumVol_x'] + volume3['sumVol_y'] / volume3['sumLen_x']
    volume3['2per_area'] = volume3['sumVol_x'] + volume3['sumVol_y'] / volume3['sumArea_x']

    return volume3.fillna(0)


def getLinkTimeFeature(df):
    mydf = deepcopy(df)
    mean = mydf.describe().travel_time['mean']
    std = mydf.describe().travel_time['std']

    mydf['mean'] = mean
    mydf['std'] = std

    filtered = mydf[abs(mydf['travel_time'] - mydf['mean']) <= 5 * mydf['std']].drop(['mean', 'std'], axis=1)

    new = filtered.groupby(['date', 'hour', 'intersection_id', 'tollgate_id', 'link']).mean().reset_index()
    new = new.pivot_table(index=['date', 'hour', 'intersection_id', 'tollgate_id'], columns='link',
                          values='travel_time') \
        .reset_index().fillna(0)

    colName = ['linkTime_' + str(i) for i in range(100, 124)]
    colName = ['date', 'hour', 'intersection_id', 'tollgate_id'] + colName
    new.columns = colName
    new2 = deepcopy(new)
    new2['hour'] = new2['hour'] - 1
    new2 = pd.merge(new, new2, on=['date', 'hour', 'intersection_id', 'tollgate_id'], how='outer').fillna(0)
    new2['hour'] = new2['hour'] + 2

    filtered2 = deepcopy(filtered)
    filtered2['hour'] = filtered2['hour'] - 1
    filtered3 = filtered.append(filtered2)
    new3 = filtered3.groupby(['date', 'hour', 'intersection_id', 'tollgate_id', 'link']).mean().reset_index()
    new3 = new3.pivot_table(index=['date', 'hour', 'intersection_id', 'tollgate_id'], columns='link',
                            values='travel_time') \
        .reset_index().fillna(0)
    colName = ['linkTime2_' + str(i) for i in range(100, 124)]
    colName = ['date', 'hour', 'intersection_id', 'tollgate_id'] + colName
    new3.columns = colName
    new3['hour'] = new3['hour'] + 2
    new2 = pd.merge(new2, new3, on=['date', 'hour', 'intersection_id', 'tollgate_id'], how='left')
    return new2

