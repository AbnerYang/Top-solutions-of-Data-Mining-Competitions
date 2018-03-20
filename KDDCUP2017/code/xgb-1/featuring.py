# -*- encoding:utf-8 -*-
from __future__ import division
import pandas as pd
import warnings
from utils import *
warnings.filterwarnings("ignore")

PHASE1_TRAIN_PATH = '../../data/dataSets/training/'
PHASE1_TEST_PATH =  '../../data/dataSets/test_1/'
PHASE2_PATH = '../../data/dataSets/test_2/'
FEATURES_PATH = '../../feature/xgb-1/'

###Load data
print 'Loading raw data...'
df1 = pd.read_csv(PHASE1_TRAIN_PATH + 'trajectories(table 5)_training.csv')
df2 = pd.read_csv(PHASE1_TEST_PATH + 'trajectories(table_5)_training2.csv')
raw_test = pd.read_csv(PHASE2_PATH + 'trajectories(table 5)_test2.csv')


### Get time information
mydf1 = translateTime(df1,'starting_time')
mydf2 = translateTime(df2,'starting_time')

###Filter samples
print 'Filtering data...'
mydf1=mydf1[(mydf1.hour>5)&(mydf1.hour<21)]
mydf2=mydf2[(mydf2.hour>5)&(mydf2.hour<21)]
mydf3 = outlierFilter(mydf1)
mydf4 = outlierFilter(mydf2)

###Get filtered raw data
train_raw = mydf3.append(mydf4)
test_raw = translateTime(raw_test,'starting_time')

###Get trainset label
print 'Generating train labels...'
mydf1 = aggregateTravelTime(train_raw,20)
mydf1 = getTimeWindows(mydf1)
mydf1 = transformLongTimeSeries(mydf1)
mydf1 = generateShortTimeSeries(mydf1,20,range(8,20),method = 'label')
label = mydf1.dropna(axis=0)

### load weather feature
weather1 = pd.read_csv(PHASE1_TRAIN_PATH + 'weather (table 7)_training_update.csv')
weather2 = pd.read_csv(PHASE1_TEST_PATH + 'weather (table 7)_test1.csv')
weather = weather1.append(weather2)
weather3 = pd.read_csv(PHASE2_PATH + 'weather (table 7)_2.csv')

indexName = ['date','tollgate_id','intersection_id','hour']

#init time stamps
timeStamps = [10,15,20,30]

#get common train feature set
print 'Getting common stat feature...'
trainFeature1 = getCommonFeature(train_raw,indexName,timeStamps)
testFeature1=getCommonFeature(test_raw,indexName,timeStamps,setup='test')
print 'Getting weather1 feature...'
trainFeature = getWeatherFeature(trainFeature1,weather)
testFeature = getWeatherFeature(testFeature1,weather3)

print 'Generating train & test feature set...'
train = getTrainSet(trainFeature,label)
train = train.dropna(axis=0)
test = getTestSet(testFeature)
print test.shape
print train.shape

print 'Writing common stat feature...'
train.to_csv(FEATURES_PATH + 'trainFeature_stat.csv',index=0)
test.to_csv(FEATURES_PATH + 'testFeature_stat.csv',index=0)

# get weather discretization feature
print 'Getting weather2 feature...'
weather = weather.append(weather3)
weather = getDate(weather)
weather['wind_speed'] = (weather.wind_speed*10).astype(int)

wind_bins = [-1,2,15,33,54,79,107,138]
wind_labels = [0,1,2,3,4,5,6]
weather['wind_level'] = pd.cut(weather['wind_speed'],bins=wind_bins,labels=wind_labels)

temp_bins = range(5,45,5)
temp_labels = range(len(temp_bins)-1)
weather['temp_level'] = pd.cut(weather['temperature'],bins=temp_bins,labels=temp_labels)

hum_bins = range(20,105,5)
hum_labels = range(len(hum_bins)-1)
weather['hum_level'] = pd.cut(weather['rel_humidity'],bins=hum_bins,labels=hum_labels)

w1 = deepcopy(weather)
w2 = deepcopy(weather)
w1['hour'] = (w1['hour']+1)%24
w2['hour'] = (w2['hour']-1)%24
weather = weather.append(w1)
weather = weather.append(w2)
weather = weather.drop(['pressure','sea_pressure','wind_direction','wind_speed','temperature','rel_humidity','precipitation'],axis=1)
print 'Writing weather2 feature...'
weather.to_csv(FEATURES_PATH + 'weatherFeat.csv',index=0)


###Get volume feature
print 'Getting volume feature..'
routes = pd.read_csv(PHASE1_TRAIN_PATH + 'routes (table 4).csv')
#train part
df = LinkseqFilter(train_raw,routes)
mydf = aggregateTravelVolume(df,20)
mydf = getTime(mydf)
pio=mydf.pivot_table(index=['date','tollgate_id','intersection_id'],columns='time',values='volume').reset_index().fillna(0)
mydf = getVolumeTimeSeries(pio,setup='train')
mydf = getVolumeFeatures(mydf)
print 'Writing train volume feature...'
mydf.to_csv(FEATURES_PATH + 'time_volume_features_train.csv',index=0)

#test part
df = LinkseqFilter(test_raw,routes)
mydf = aggregateTravelVolume(df,20)
mydf = getTime(mydf)
pio=mydf.pivot_table(index=['date','tollgate_id','intersection_id'],columns='time',values='volume').reset_index().fillna(0)
mydf = getVolumeTimeSeries(pio,setup='test')
mydf = getVolumeFeatures(mydf)
print 'Writing test volume feature...'
mydf.to_csv(FEATURES_PATH + 'time_volume_features_test.csv',index=0)

###Get link feature
print 'Getting link feature...'
links = pd.read_csv(PHASE1_TRAIN_PATH + 'links (table 3).csv')
link_dic = {}
ldata = links.values
for row in ldata:
    link_dic[row[0]] = {'length':row[1],'lanes':row[2],'area':row[1]*row[2]}

mydf = getLink(routes,links)
mydf = getLinkFeature(link_dic,mydf)
print 'Writing link feature...'
mydf.to_csv(FEATURES_PATH + 'links_features.csv',index=0)

###Get link-time/volume feature
print 'Getting link-time/volume feature...'
df = train_raw.append(test_raw)
mydf = splitSeq(df)
mydf = splitTime(mydf)
linkVolume = getLinkVolumeFeature(mydf,link_dic)
linkTime = getLinkTimeFeature(mydf)
linkSpeed = getLinkSpeedFeature(mydf,link_dic)

print 'Writing link-time/volume feature...'
linkVolume.to_csv(FEATURES_PATH + 'linkVolume.csv',index=0)
linkTime.to_csv(FEATURES_PATH + 'linkTime.csv',index=0)
linkSpeed.to_csv(FEATURES_PATH + 'linkSpeed.csv',index=0)




