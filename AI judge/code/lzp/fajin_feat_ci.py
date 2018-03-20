# -*-encoding:utf-8 -*-
import pandas as pd

import numpy as np
import re

input_train_path = '../../data/train.txt'
input_test_path = '../../data/test.txt'

train1_path = '../../feature/lzp/fajin.train.ci1.txt'
test1_path = '../../feature/lzp/fajin.test.ci1.txt'

stop_path = '../../data/stop.txt'

train2_path = '../../feature/lzp/fajin.train.ci2.txt'
test2_path = '../../feature/lzp/fajin.test.ci2.txt'

id_path = '../../feature/lzp/fajin.ci.id.tsv'

cnt_thre = 4

def get_word_id():
    df = pd.read_csv(train1_path, sep='\t', header=None, encoding='utf8')
    
    X = df.values
    dic = {}

    for i in range(len(X)):
        if i%1000 == 0:
            print (i)
            
        x = X[i]
        se = set()
        xx = x[1].split(' ')
        #for z in x[1]:
        for z in xx:
            if z in se:
                continue
            se.add(z)
            
            if z not in dic:
                dic[z] = 0
            dic[z] += 1

    T = []
    for k, v in dic.items():
        if v > cnt_thre:
            T.append([k, v])
    df = pd.DataFrame(T)
    df.to_csv(id_path, index=False, header=None, encoding='utf8')
    

def read_dic():
    df = pd.read_csv(id_path, header=None, encoding='utf8')
    dic = {}
    i = 1
    for x in df.values:
        dic[x[0]] = i
        i += 1
    return dic, i
    
    
def cnn_feature(flag, nrows=None):
    outpath = ''
    if flag == 'train':
        df = pd.read_csv(train1_path, sep='\t', header=None, encoding='utf8', nrows=None)
        outpath = train2_path
    else:
        df = pd.read_csv(test1_path, sep='\t', header=None, encoding='utf8', nrows=None)
        outpath = test2_path
        
    dic, max_id = read_dic()
    
    T = []
    X = df.values
    for i in range(len(X)):
        if i%1000 == 0:
            print (i)
#        if i == 2000:
#            break
            
        x = X[i]
        zz = x[1].split(' ')
        t = [x[0], -1]
        if flag == 'train':
            t = [x[0], x[2]]
       
        #if len(x[1]) > 10000:
        if len(zz) > 1000:
            zz = zz[:1000]

        sn = []
        #for z in x[1]:
        for z in zz:
            if z in dic:
                sn.append(dic[z])
            else:
                sn.append(max_id)
        t = t + sn
        T.append(t)
    df = pd.DataFrame(T)
    #df = df.astype('int')
    df.to_csv(outpath, index=False, header=None, encoding='utf8')
    
def get_alpha(number, flag=False):
    number = float(number.replace(',','').replace('，',''))
    if flag:
        number *= 10000
        
#    list1 = [1000, 2000, 3000, 4000, 5000, 10000, 500000]
#    list2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    list1 = [30, 100, 300, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 50000, 100000, 500000]
    list2 = ['QA', 'QB', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QJ', 'QK', 'QL', 'QM', 'QN', 'QO']

    i = 0
    while i<len(list1):
        if number<list1[i]:
            break
        
        i += 1
            
    return list2[i]

    
def replace_money(string):
    string = string.encode('utf-8')
    string = string.replace('余元', '元').replace('万余元', '万元').replace('余万元', '万元')
    r = re.compile('(\\d+((,\\d+)|(，\\d+))*(\.\\d+)?)元')
    numbers = r.findall(string)
    
    for number in numbers:
        number = number[0]
        alpha = get_alpha(number)
        string = string.replace(number, alpha)
        
    r = re.compile('(\\d+((,\\d+)|(，\\d+))*(\.\\d+)?)万元')
    numbers = r.findall(string)
    
    for number in numbers:
        number = number[0]
        alpha = get_alpha(number, True)
        string = string.replace(number, alpha).replace('万元','元')
        
    return string
        

import jieba,codecs
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

from tqdm import tqdm    
def replace_train_test(nrows=None):
    files = [input_test_path, input_train_path]
    files1 = [test1_path, train1_path]

    stopwords = {}
    for line in codecs.open(stop_path, 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1
                  
    for i in range(len(files)):
        print (files[i])
        df = pd.read_csv(files[i], sep='\t', header=None, encoding='utf8', nrows=nrows)
    
        X = df.values
        for j in tqdm(range(len(X))):
            if len(X[j][1]) > 0:
                X[j][1] = replace_money(X[j][1])
                X[j][1] = splitWord(X[j][1],stopwords).decode('utf8')
        df = pd.DataFrame(X, columns=df.columns)
        print (files1[i])
        df.to_csv(files1[i], sep='\t', index=False, header=False, encoding='utf8')
        
    

def run(nrows=None):
    replace_train_test(nrows) #替换金额类数值， 分词
    
    global cnt_thre
    cnt_thre = 6
    get_word_id() #给词编id
    
    cnn_feature('train', nrows) #获取训练集的特征
    cnn_feature('test', nrows) #获取测试集的特征
    
        
def main():   
    replace_train_test() #替换金额类数值， 分词
    
    global cnt_thre
    cnt_thre = 6
    get_word_id() #给词编id
    
    cnn_feature('train') #获取训练集的特征
    cnn_feature('test') #获取测试集的特征
    

    print ('tt')
    
if __name__ == '__main__':
    main()