#-*-coding:utf-8-*- 




# %%----------------------------------------------- import---------------------------------------
from collections import defaultdict
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.models.lsimodel import LsiModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import jieba.posseg
import jieba.analyse
import time
import re
import codecs
import xlrd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import seaborn as sns

from sklearn.cross_validation import train_test_split

import os

import copy

debugflag = False


def debug(stri):
    if debugflag:
        print stri
        
pwd =os.getcwd()


#base_path = os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")

data_path='../../data/'


#code_path = base_path+u'/code/'

feature_path= '../../feature/ljh/'



# %%-------------------------------------------------  split word -------------------------------------------

def splitWord(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result.encode('utf-8')

def splitword(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result

 
def filterword(x):
    return x.replace(u'万万元',u'万元').replace(u'万多万元',u'万元').replace(u'万余万元',u'万元').replace(u'十万八万元',u'八万元').replace(u'一万两万元',u'一万元').replace(u'一万多两万元',u'两万元')   



def filter():
    train = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    test = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')
    train['text'] = train['text'].map(filterword)
    test['text'] = test['text'].map(filterword)
    train.to_csv(data_path + 'train_content.csv', index=False, encoding='utf-8')
    test.to_csv(data_path + 'test_content.csv', index=False, encoding='utf-8')


def preprocess(mode='train'):
    stopwords = {}
    for line in codecs.open(data_path + 'stop.txt', 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1
    if mode == 'train':
        data = pd.read_csv(data_path + 'train.txt', delimiter='\t', header=None,
                           names=('id', 'text', 'penalty', 'laws'), encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test.txt', delimiter='\t', header=None, names=('id', 'text'),
                           encoding='utf-8')
    if mode == 'law':
        data = pd.read_csv(data_path + 'form-laws.txt', delimiter='\t', header=None, names=('id', 'text'),
                           encoding='utf-8')
        data['prefix_text'] = data['text'].map(lambda x: x[x.index(u'【') + 1:x.index(u'】')])
        data['prefix_text'] = data['prefix_text'].map(lambda x: splitWord(x, stopwords))
    data['doc'] = data['text'].map(lambda x: splitWord(x, stopwords))
    data['text'] = data['text'].map(filterword)
    data.to_csv(data_path + '%s_content.csv' % (mode), index=False, encoding='utf-8')







# %%----------------------------------get num feature---------------------------------------------------


def getweight(x):
    r1 = re.compile(u'[0-9]\d*\.?\d*克')
    r2 = re.compile(u'[0-9]\d*\.?\d*g')
    r3 = re.compile(u'[0-9]\d*\.?\d*mg')
    r4 = re.compile(u'[0-9]\d*\.?\d*千克')
    w1 = r1.findall(x)
    w2 = r2.findall(x)
    w3 = r3.findall(x)
    w4 = r4.findall(x)
    total = 0
    for w in w1:
        w = float(w.replace(u'克', ''))
        total = total + w
    for w in w2:
        w = float(w.replace('g', ''))
        total = total + w
    for w in w3:
        w = float(w.replace('mg', ''))
        total = total + w / 1000
    for w in w4:
        w = float(w.replace(u'千克', ''))
        total = total + w * 1000
    return total



def getsquare(x):
    r1 = re.compile(u'[0-9]\d*\.?\d*立方米')
    mon1 = r1.findall(x)
    total = 0
    for mon in mon1:
        mon = float(mon.replace(u'立方米', ''))
        total = total + mon
    return total


def gettree(x):
    r1 = re.compile(u'[0-9]\d*\.?\d*株')
    mon1 = r1.findall(x)
    total = 0
    for mon in mon1:
        mon = float(mon.replace(u'株', ''))
        total = total + mon
    return total


# %% ------------------------------- law one hot ---------------------------------------



def getmoney_prefix_file(arr):
    
    r1 = re.compile(u'(.{7})\d[\d，]*(?:\.\d+)?[余多]?元')
    r2 = re.compile(u'(.{7})(?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万][余多]?元')
    r3 = re.compile(u'(.{7})[一二两三零四五六七八九万千百十][余多]?[一二两三零四五六七八九万千百十]+[余多]?元')
#
#    r1=re.compile(u'(.{7})(?:\d+，?)?\d+(?:.\d+)?元')
#    r2=re.compile(u'(.{7})(?:\d+，?)?\d+(?:.\d+)?余元')
#    r3=re.compile(u'(.{7})(?:\d+，?)?\d+(?:.\d+)?万元')
#    r4=re.compile(u'(.{7})(?:\d+，?)?\d+(?:.\d+)?余万元')
    
    for  x in arr:
        mon1 = r1.findall(x)
        mon2 = r2.findall(x)
        mon3 = r3.findall(x)
        mon1="\n".join(mon1)
        mon2="\n".join(mon2)
        mon3="\n".join(mon3)
        lawfile=codecs.open(data_path+u'mon.txt','a','utf-8')
        lawfile.write(mon1)
        lawfile.write(mon2)
        lawfile.write(mon3)
        
        
        
def getmoneykeyword_file():
    stopwords = {}
    for line in codecs.open(data_path+'stop.txt','r','utf-8'):
        stopwords[line.rstrip()]=1
    keydict ={}
    for line in codecs.open(data_path+u'mon.txt','r','utf-8'):
        keylist=splitword(line.rstrip(),stopwords)
        keylist = keylist.split(' ')
        for x in keylist:
            if x in keydict:
                keydict[x]=keydict[x]+1
            else:
                keydict[x]=1
    
    
    lawkeydictsorted=sorted(keydict.iteritems(), key=lambda d:d[1],reverse=True)
    
    lawkeys=[]
    
    r =re.compile('\d')
    
    for lawkey in  lawkeydictsorted:
        if lawkey[1]>1000 and len(lawkey[0])>1 and u'某' not in lawkey[0] and len(r.findall(lawkey[0]))==0:
            lawkeys.append(lawkey[0])
    
    content = '\n'.join(lawkeys)
    lawfile=codecs.open(data_path+u'monkey.txt','w','utf-8')
    
    lawfile.write(content)




def getlaw1(arr, i=100):
    r = re.compile(u'犯.{2,10}罪')
    law_dict = {}
    for x in arr:
        lawlist = r.findall(x)
        for law in lawlist:
            law = law.replace(u'犯', '').replace(u'罪', '')
            if law not in law_dict:
                law_dict[law] = 1
            else:
                law_dict[law] = law_dict[law] + 1

    law_list = sorted(law_dict.iteritems(), key=lambda d: d[1], reverse=True)

    law_form_data = pd.read_csv(data_path + '/1-train/form-laws.txt', delimiter='\t', header=None, names=('id', 'text'),
                                encoding='utf-8')
    law_from = law_form_data[101:448]['text'].map(lambda x: x[x.index(u'【') + 1:x.index(u'】') - 1]).values
    law_from = ' '.join(law_from)
    laws = []
    for law in law_list[0:i]:
        if law[0] in law_from:
            laws.append(law[0])

    content = "\n".join(laws)

    lawfile = codecs.open(data_path + 'law_list.txt', 'w', 'utf-8')

    lawfile.write(content)


def getlawkey(arr):
    law_form_data = pd.read_csv(data_path + 'law_content.csv', encoding='utf-8')
    # law_from = law_data[0:101]['text'].map(lambda x: x[x.index(u'【')+1:x.index(u'】')]).values

    lawkeydict = {}
    lawkeylist = []
    law_from = law_form_data['prefix_text'][0:101].values

    for law in law_from:
        if pd.isnull(law):
            continue
        templist = law.split(' ')
        for temp in templist:
            if temp not in lawkeylist:
                lawkeylist.append(temp)

    for x in arr:
        for lawkey in lawkeylist:
            if lawkey in x:
                if lawkey in lawkeydict:
                    lawkeydict[lawkey] = lawkeydict[lawkey] + 1
                else:
                    lawkeydict[lawkey] = 0

    lawkeydictsorted = sorted(lawkeydict.iteritems(), key=lambda d: d[1], reverse=True)

    lawkeys = []

    for lawkey in lawkeydictsorted:
        lawkeys.append(lawkey[0])

    content = '\n'.join(lawkeys)
    lawfile = codecs.open(data_path + 'law_key_list.txt', 'w', 'utf-8')

    lawfile.write(content)


common_used_numerals = {u'零': 0, u'一': 1, u'二': 2, u'两': 2, u'三': 3, u'四': 4, u'五': 5,
                        u'六': 6, u'七': 7, u'八': 8, u'九': 9, u'十': 10, u'百': 100, u'千': 1000, u'万': 10000,
                        u'亿': 100000000}


def subFunc(uchars_chinese):
    try:
        total = 0
        r = 1  # 表示单位：个十百千...
        for i in range(len(uchars_chinese) - 1, -1, -1):
            val = common_used_numerals.get(uchars_chinese[i])
            if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                if val > r:
                    r = val
                    total = total + val
                else:
                    r = r * val
                    # total =total + r * x
            elif val >= 10:
                if val > r:
                    r = val
                else:
                    r = r * val
            else:
                total = total + r * val
        return total
    except:
        return 0.0


def chinese2digits(chinese):
    p2 = chinese
    p11, p12 = '', ''
    if u'万' in p2:
        p12, p2 = chinese.split(u'万')
    if u'亿' in p12:
        p11, p12 = p12.split(u'亿')
    ps = [p11, p12, p2]
    ms = [0.0, 0.0, 0.0]
    #    print 'call1'
    for i, p in enumerate(ps):
        ms[i] = subFunc(p)
    # print 'return1'
    return ms[0] * 100000000 + ms[1] * 10000 + ms[2]






r1 = re.compile(u'(\d[\d，]*(?:\.\d+)?)[余多]?元')
r2 = re.compile(u'((?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万])[余多]?元')
r3 = re.compile(u'([一二两三零四五六七八九万千百十][多余]?[一二两三零四五六七八九万千百十]+[余多]?)元')

r4 = re.compile(u'(?:一共|共计|合计|总价值|总计|累计|总共|总额|总金额)[^\d]{0,7}(\d[\d，]*(?:\.\d+)?)[余多]?元')
r5 = re.compile(u'(?:一共|共计|合计|总价值|总计|累计|总共|总额|总金额)[^\d]{0,7}((?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万])[余多]?元')
r6 = re.compile(u'(?:一共|共计|合计|总价值|总计|累计|总共|总额|总金额)[^\d]{0,7}([一二两三零四五六七八九万千百十][余多]?[一二两三零四五六七八九万千百十]+[余多]?)元')

r7 = re.compile(u'(?:海洛因|毒资|冰毒|毒品|手机|轿车|摩托车|香烟|电脑|电动车|动车|车辆|项链|电瓶|香烟|包内|钱包|抽屉|身上|家中|工资|被盗|赃款|获利|盗走|窃得|涉案|销赃|赌资|诈骗|盗得|骗得|渔利|赃物|窃取|戒指|自行车|抢走|赌博|手表|电缆线|骗走)[^\d]{0,7}(\d[\d，]*(?:\.\d+)?)[余多]?元')
r8 = re.compile(u'(?:海洛因|毒资|冰毒|毒品|手机|轿车|摩托车|香烟|电脑|电动车|动车|车辆|项链|电瓶|香烟|包内|钱包|抽屉|身上|家中|工资|被盗|赃款|获利|盗走|窃得|涉案|销赃|赌资|诈骗|盗得|骗得|渔利|赃物|窃取|戒指|自行车|抢走|赌博|手表|电缆线|骗走)[^\d]{0,7}((?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万])[余多]?元')
r9 = re.compile(u'(?:海洛因|毒资|冰毒|毒品|手机|轿车|摩托车|香烟|电脑|电动车|动车|车辆|项链|电瓶|香烟|包内|钱包|抽屉|身上|家中|工资|被盗|赃款|获利|盗走|窃得|涉案|销赃|赌资|诈骗|盗得|骗得|渔利|赃物|窃取|戒指|自行车|抢走|赌博|手表|电缆线|骗走)[^\d]{0,7}([一二两三零四五六七八九万千百十][余多]?[一二两三零四五六七八九万千百十]+[余多]?)元')

r10 = re.compile(u'(?:支付|利息|本金|所得|存款|透支|资金|贷款|账户|经济损失|转账|汇款|银行|抽头|投资|取款|借款|退赔|索要|还款|送给|赔偿|借给|损失|购买|公司|收取|归还|退还|缴纳|消费|出资|额度|税款|欠款|税额|合同|价税|偿还)[^\d]{0,7}(\d[\d，]*(?:\.\d+)?)[余多]?元')
r11 = re.compile(u'(?:支付|利息|本金|所得|存款|透支|资金|贷款|账户|经济损失|转账|汇款|银行|抽头|投资|取款|借款|退赔|索要|还款|送给|赔偿|借给|损失|购买|公司|收取|归还|退还|缴纳|消费|出资|额度|税款|欠款|税额|合同|价税|偿还)[^\d]{0,7}v((?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万])[余多]?元')
r12 = re.compile(u'(?:支付|利息|本金|所得|存款|透支|资金|贷款|账户|经济损失|转账|汇款|银行|抽头|投资|取款|借款|退赔|索要|还款|送给|赔偿|借给|损失|购买|公司|收取|归还|退还|缴纳|消费|出资|额度|税款|欠款|税额|合同|价税|偿还)[^\d]{0,7}([一二两三零四五六七八九万千百十][余多]?[一二两三零四五六七八九万千百十]+[余多]?)元')

r13 = re.compile(u'(?:罚金|罚款|保证金)[^\d]{0,7}(\d[\d，]*(?:\.\d+)?)[余多]?元')
r14 = re.compile(u'(?:罚金|罚款|保证金)[^\d]{0,7}((?:\d[\d，]*(?:\.\d+)?)[余多]?[十百千万])[余多]?元')
r15 = re.compile(u'(?:罚金|罚款|保证金)[^\d]{0,7}([一二两三零四五六七八九万千百十][余多]?[一二两三零四五六七八九万千百十]+[余多]?)元')

regulardict = {'all': [r1, r2, r3], 'total': [r4, r5, r6], 'fanzui': [r7, r8, r9], 'jinron': [r10, r11, r12],
               'fakuan': [r13, r14, r15]}
keylist = ['all', 'total', 'fanzui', 'jinron', 'fakuan']






def getkeymoney(x):
    mondict = {}
    for res in regulardict:

        mon1 = regulardict[res][0].findall(x)
        mon2 = regulardict[res][1].findall(x)
        mon3 = regulardict[res][2].findall(x)
        monlist = []

        debug('mon1 start')
        debug(mon1)

        for mon in mon1:
            debug('||' + mon)
            temp = mon
            temp = float(temp.replace(u'，', '').replace(u'余', '').replace(u'多', ''))
            debug(temp)
            monlist.append(temp)

        #
        #
        debug('mon2 start')
        debug(mon2)
        for mon in mon2:
            debug('||' + mon)
            unitpow = 0
            temp = mon
            if u'十' in mon:
                unitpow = 10
                temp = temp.replace(u'十', '')
            if u'百' in mon:
                unitpow = 100
                temp = temp.replace(u'百', '')
            if u'千' in mon:
                unitpow = 1000
                temp = temp.replace(u'千', '')
            if u'万' in mon:
                unitpow = 10000
                temp = temp.replace(u'万', '')
            temp = temp.replace(u'，', '')
            temp = temp.replace(u'余', '')
            temp = temp.replace(u'多', '')

            temp = float(temp) * unitpow

            debug(temp)
            monlist.append(temp)

        debug('mon3 start')
        debug(mon3)
        for mon in mon3:
            debug('||' + mon)
            temp = mon
            temp = chinese2digits((temp.replace(u'，', '').replace(u'余', '').replace(u'多', '')))
            debug(temp)
            monlist.append(temp)

        mondict[res] = monlist

    return mondict




def getmon_dict(x, key, func):
    if key in x:
        if len(x[key]) == 0:
            return 0
        return func(x[key])
    else:
        return 0


def getmon_list(x, func):
    if len(x) > 0:
        return func(x)
    else:
        return 0
    

def getkeymon(data):
    LogInfo('start')
    data['mon_dict'] = data['text'].map(getkeymoney)

    LogInfo('all start')
    for key in keylist:
        data[key + '_sum'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.sum))
        data[key + '_mean'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.mean))
        data[key + '_max'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.max))
        data[key + '_min'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.min))
        data[key + '_std'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.std))
        data[key + '_ptp'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.ptp))
        data[key + '_median'] = data['mon_dict'].map(lambda x: getmon_dict(x, key, np.median))

    LogInfo('all end')

    data.drop(['mon_dict'], axis=1, inplace=True)


def getlaw_key_onehot(data):
    law_list = []
    for line in codecs.open(data_path + 'law_key_list.txt', 'r', 'utf-8'):
        law_list.append(line.rstrip())
    for i in range(0, len(law_list)):
        ri = re.compile(law_list[i].rstrip(''))
        data['lawkey_' + str(i)] = data['text'].map(lambda x: len(ri.findall(x)))


def gettfidf_random(data):
    tfidf = pd.read_csv(data_path + 'tf_all.csv', encoding='utf-8')
    part1 = []
    part2 = []
    part3 = []
    while len(part1) < 600:
        tmp = np.random.randint(0, 1000)
        if tmp not in part1:
            part1.append(tmp)
    while len(part2) < 300:
        tmp = np.random.randint(1000, 2000)
        if tmp not in part2:
            part2.append(tmp)
    while len(part3) < 100:
        tmp = np.random.randint(2000, 3000)
        if tmp not in part3:
            part3.append(tmp)
    part = part1 + part2 + part3
    word_list = tfidf['word'][part].values
    for i in range(0, len(word_list)):
        ri = re.compile(word_list[i].rstrip(''))
        data['tfidf_' + word_list[i]] = data['doc'].map(lambda x: len(ri.findall(x)))


def gettfidf_onehot(data, num):
    tfidf = pd.read_csv(data_path + 'tf_all.csv', encoding='utf-8')
    word_list = tfidf['word'][0:num + 1].values
    for i in range(0, len(word_list)):
        ri = re.compile(word_list[i].rstrip(''))
        data['tfidf_' + word_list[i]] = data['doc'].map(lambda x: len(ri.findall(x)))


def getlaw_onehot(data):
    law_list = []
    for line in codecs.open(data_path + 'law_list_new.txt', 'r', 'utf-8'):
        law_list.append(line.rstrip())
    data['law_sum'] = 0
    print law_list
    for i in range(0, len(law_list)):
        ri = re.compile(law_list[i])
        print i, law_list[i]
        data['law_' + str(i)] = data['text'].map(lambda x: len(ri.findall(x)))
        data['law_sum'] = data['law_sum'] + data['law_' + str(i)]








# %% -------------------------------line num ;yuan;district ----------------------------------------------

def getlinenum(x):
    return x.count(',') + x.count('.') + x.count(';') + x.count(u'，') + x.count(u'。') + x.count(u'；')


def getyuan(s):
    if s[0] == u'原':
        return 1
    if s[0] == u'罪':
        return 2
    return 0


def getdocprovince(x):
    # x= codecs.utf_8_decode(x)[0]
    x = x.split(' ')
    i = 0
    while i < 7 and len(x) > i:

        if x[i].find(u'省') != -1 or x[i].find(u'自治区') != -1:
            return x[i]
        i = i + 1
    return " "


def getdoccity(x):
    # x= codecs.utf_8_decode(x)[0]
    x = x.split(' ')
    i = 0
    while i < 7 and len(x) > i:
        if x[i].find(u'市') != -1:
            return x[i]
        i = i + 1
    return " "


def getdocdistrict(x):
    # x= codecs.utf_8_decode(x)[0]
    x = x.split(' ')
    i = 0
    while i < 7 and len(x) > i:
        if x[i].find(u'县') != -1:
            return x[i]
        if x[i].find(u'区') != -1 and x[i].find(u'自治区') == -1:
            return x[i]
        i = i + 1

    return " "


def getcourtlevel(x):
    if (not pd.isnull(x[2])) and x[2] != "" and x[2] != " ":
        return 0

    if (not pd.isnull(x[1])) and x[1] != "" and x[1] != " ":
        return 1

    if (not pd.isnull(x[0])) and x[0] != "" and x[0] != " ":
        return 2

    return 3


# %%-----------------------------------get feature--------------------------------------------

def getcontent():
    preprocess('train')
    preprocess('test')
    preprocess('law')






map_code = pd.read_excel(data_path + 'map.xls')


def getprovince_code(x):
    tmpstr = ''

    if (not pd.isnull(x[2])) and x[2] != "" and x[2] != " ":
        tmpstr = x[2]

    if (not pd.isnull(x[1])) and x[1] != "" and x[1] != " ":
        tmpstr = x[1]

    if (not pd.isnull(x[0])) and x[0] != "" and x[0] != " ":
        tmpstr = x[0]

    if tmpstr == '':
        return 0

    debug(tmpstr)
    for tmp in map_code.values:
        if tmpstr in tmp[1]:
            debug(tmpstr)
            return str(tmp[0])[0:2]
    return 0


def getprovince_onehot():
    train = pd.read_csv(data_path + 'train_base.csv', encoding='utf-8')
    test = pd.read_csv(data_path + 'test_base.csv', encoding='utf-8')
    tmp = np.append(train.province_code.values, test.province_code)

    tmp = pd.get_dummies(tmp, prefix='province')

    train_temp = tmp[0:120000].reset_index().drop('index', axis=1)
    test_temp = tmp[120000:210000].reset_index().drop('index', axis=1)

    train = pd.concat([train, train_temp], axis=1)
    test = pd.concat([test, test_temp], axis=1)
    #
    #    train =train.drop('province_code',axis=1)
    #    test=test.drop('province_code',axis=1)
    #
    train.to_csv(data_path + 'train_baseafter.csv', encoding='utf-8', index=None)

    test.to_csv(data_path + 'test_baseafter.csv', encoding='utf-8', index=None)


def getbasefeature(mode='train'):
    if mode == 'train':
        data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')

    data['doc_len'] = data['doc'].map(lambda x: len(x))
    data['linenum'] = data['text'].map(getlinenum)
    data['is_contain_yuan'] = data['text'].map(getyuan)

    data['province'] = data['doc'].map(getdocprovince)

    data['city'] = data['doc'].map(getdoccity)
    data['district'] = data['doc'].map(getdocdistrict)

    data['courtlevel'] = data[['province', 'city', 'district']].apply(getcourtlevel, axis=1)

    data['province'] = data['province'].map(
        lambda x: x.replace(u'自治区', '').replace(u'回族', '').replace(u'壮族', '').replace(u'维吾尔', ''))
    data['city'] = data['city'].map(
        lambda x: x.replace(u'自治区', '').replace(u'回族', '').replace(u'壮族', '').replace(u'维吾尔', ''))
    data['district'] = data['district'].map(
        lambda x: x.replace(u'自治区', '').replace(u'回族', '').replace(u'壮族', '').replace(u'维吾尔', ''))

    data['province_code'] = data[['province', 'city', 'district']].apply(getprovince_code, axis=1)

    data = data.drop(['doc', 'province', 'city', 'district', 'text'], axis=1)

    if mode == 'train':
        data = data.drop(['laws', 'penalty'], axis=1)

    data.to_csv(data_path + mode + '_base.csv', encoding='utf-8', index=None)


def getnumfeature(mode='train'):
    if mode == 'train':
        data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')

    # data['money'] = data['text'].map(getmoney)

    data['weight'] = data['text'].map(getweight)
    data['tree'] = data['text'].map(gettree)
    data['square'] = data['text'].map(getsquare)

    data = data.drop(['doc', 'text'], axis=1)

    if mode == 'train':
        data = data.drop(['laws', 'penalty'], axis=1)

    data.to_csv(data_path  + mode + '_num.csv', encoding='utf-8', index=None)


def getlaw_list_file(i=100):
    data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    text_arr = data.text.values
    data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')

    text_arr = np.append(text_arr, data.text.values)
    getlaw1(text_arr, i)


def getzerocount(x):
    i = 0
    for tmp in x:
        if tmp == 0:
            i = i + 1
    return i


def get_tfidf_file():
    train_data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    documents = []
    for i in range(1, 9):
        temp = train_data['doc'][train_data.penalty == i].values
        temp = " ".join(temp)
        documents.append(temp)

    vectorizer = CountVectorizer(encoding='utf-8')  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(documents))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    temp = pd.DataFrame({'word': word})

    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        temp['weight_' + str(i + 1)] = weight[i]
    temp['std'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.std(x), axis=1)
    temp['ptp'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.ptp(x), axis=1)
    temp['mean'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.mean(x), axis=1)
    temp['max'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.max(x), axis=1)
    temp['min'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.min(x), axis=1)
    temp['median'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        lambda x: np.median(x), axis=1)

    temp['zero_count'] = temp[
        ['weight_1', 'weight_2', 'weight_3', 'weight_4', 'weight_5', 'weight_6', 'weight_7', 'weight_8']].apply(
        getzerocount, axis=1)
    temp = temp[temp['zero_count'] <= 4]

    temp = temp.sort_values(by=['std', 'max', 'median', 'ptp'], ascending=False)
    temp.to_csv(data_path + 'tf_all.csv', index=None, encoding='utf-8')


## 再手动过滤了一部分
def getlaw_key_list_file():
    data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    text_arr = data.text.values
    data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')

    text_arr = np.append(text_arr, data.text.values)
    getlawkey(text_arr)


def gettdidfonehotfeature(mode='train', num=500):
    if mode == 'train':
        data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')
    gettfidf_onehot(data, num)
    data = data.drop(['doc', 'text'], axis=1)

    if mode == 'train':
        data = data.drop(['laws', 'penalty'], axis=1)

    data.to_csv(data_path  + mode + '_tfidf_'+str(num)+'.csv', encoding='utf-8', index=None)


def gettdidf_random_onehotfeature(mode='train'):
    if mode == 'train':
        data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')
    gettfidf_random(data)
    data = data.drop(['doc', 'text'], axis=1)

    if mode == 'train':
        data = data.drop(['laws', 'penalty'], axis=1)

    data.to_csv(data_path  + mode + '_tfidf.csv', encoding='utf-8', index=None)



def getkeymonfeature(mode='train'):
    if mode == 'train':
        data = pd.read_csv(data_path + 'train_content.csv', encoding='utf-8')
    if mode == 'test':
        data = pd.read_csv(data_path + 'test_content.csv', encoding='utf-8')
    getkeymon(data)

    data = data.drop(['doc', 'text'], axis=1)
    if mode == 'train':
        data = data.drop(['laws', 'penalty'], axis=1)
    data.to_csv(data_path  + mode + '_keymontotal.csv', encoding='utf-8', index=None)


def LogInfo(stri):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print str(now) + ' ' + str(stri)



    
def combineFeature():
    
    
    train = pd.read_csv(data_path+'train_content.csv',encoding='utf-8')
    test =pd.read_csv(data_path+'test_content.csv',encoding='utf-8')
    
    train = train.drop(['doc','text','laws'],axis=1)
    test = test.drop(['doc','text'],axis=1)
    
    
    feature = ['num','baseafter','keymontotal','tfidf_2000']

    
    for fea in feature:
        
            
            
        train_feature = pd.read_csv(data_path+'train_'+fea+'.csv',encoding='utf-8')
        test_feature = pd.read_csv(data_path+'test_'+fea+'.csv',encoding='utf-8')
        
        
        if 'base' in fea:
            
            train_feature =train_feature.drop('province_code',axis=1)
            test_feature=test_feature.drop('province_code',axis=1)
        
    
    
        train = pd.merge(train,train_feature,how='left',on='id')
        test = pd.merge(test,test_feature,how='left',on='id')
    train.to_csv(feature_path+'train_feature.csv',encoding='utf-8',index=None)
    test.to_csv(feature_path+'test_feature.csv',encoding='utf-8',index=None)
    
def getUseColumn(data):
    d = copy.deepcopy(data)
    k = d.var()
    # print k
    print (k[(k == np.nan) | (k == 0)].index.values)
    col1 = k[(k != np.nan) & (k != 0)].index.values
    return col1


def get_diff_num_feature(num):
    train = pd.read_csv(feature_path+'train_feature.csv',encoding='utf-8')
    test =pd.read_csv(feature_path+'test_feature.csv',encoding='utf-8')
    

    
    
    LogInfo(train.shape)
    LogInfo(test.shape)
 
    

    cols  = train.columns
    
    test_cols = test.columns
    
    tfidf_cols =[ True if 'tfidf_' in col else False for col in cols]
    other_cols =[ False if x else True for x in tfidf_cols]
    
    test_tfidf_cols =[ True if 'tfidf_' in col else False for col in test_cols]
    test_other_cols =[ False if x else True for x in test_tfidf_cols]
    
    tfidf = cols[tfidf_cols]
    other = cols[other_cols]
    
    test_tfidf= test_cols[test_tfidf_cols]
    test_other  = test_cols[test_other_cols]

        
    fea_col =other.append(tfidf[0:num])
    test_fea_col = test_other.append(test_tfidf[0:num])
    train_feature = train[fea_col]
    test_feature = test[test_fea_col]
    train_feature.drop(['penalty'],axis=1,inplace=True)
    train_feature.to_csv(feature_path+'train_feature_'+str(num)+'.csv',encoding='utf-8',index=None)
    test_feature.to_csv(feature_path+'test_feature_'+str(num)+'.csv',encoding='utf-8',index=None)


rule = re.compile(u'第[一二三四五六七八两九十百千万]{1,10}条')

def getrule(x):
    rulelist =[]
    rules = rule.findall(x)
    if len(rules)!=0:
        for tmp in rules:
            rulelist.append(tmp)
    return ','.join(rulelist)


def getrules(data):
    data['rules']=data['text'].map(getrule)
    
def get_rule_data():

    train = pd.read_csv(data_path + 'train.txt', delimiter='\t', header=None,names=('id', 'text', 'penalty', 'laws'), encoding='utf-8')
    test = pd.read_csv(data_path + 'test.txt', delimiter='\t', header=None, names=('id', 'text'),encoding='utf-8')
    
    getrules(train)
    getrules(test)
    
    train.drop(['text','penalty','laws'],axis=1,inplace=True)
    test.drop(['text'],axis=1,inplace=True)
    train.to_csv(feature_path+'train_rule.csv',index=None,encoding='utf-8')
    test.to_csv(feature_path+'test_rule.csv',index=None,encoding='utf-8')
    
def run():
    get_rule_data() ## 规则提取法文

    modes = ['train','test']
    
    for mode in modes:
        preprocess(mode) ##预处理，分词
    

    get_tfidf_file()##分组tf-idf选词
    
    filter()##文本清洗

    for mode in modes:
        getbasefeature(mode)  ##基本统计量特征
        getkeymonfeature(mode)  ## 金额特征
        gettdidfonehotfeature(mode,1000) ## 关键词特征
        getnumfeature(mode)## 数字类 统计量特征
    getprovince_onehot() ## 省份做one-hot
    combineFeature()## 合并各部分feature
#   get_diff_num_feature(1000)  ## 提取 1000个词的feature 用于深度学习 wide

    


    
    