# -*-encoding:utf-8 -*-
import pandas as pd

import numpy as np
import re

input_train_path = '../../data/train.txt'
input_test_path = '../../data/test.txt'

train1_path = '../../feature/lzp/fawen.train.zi1.txt'
test1_path = '../../feature/lzp/fawen.test.zi1.txt'

stop_path = '../../data/stop.txt'

train2_path = '../../feature/lzp/fawen.train.zi2.txt'
test2_path = '../../feature/lzp/fawen.test.zi2.txt'

id_path = '../../feature/lzp/fawen.zi.id.tsv'


max_seq_len = 3000

def get_word_id():
    df = pd.read_csv(train1_path, sep='\t', header=None, encoding='utf8')
    
    X = df.values
    dic = {}

    for i in range(len(X)):
        if i%1000 == 0:
            print (i)
            
        x = X[i]
        se = set()
        
        if len(x[1]) > max_seq_len:
            x[1] = x[1][:max_seq_len]

        for z in x[1]:
            if z in se:
                continue
            se.add(z)
            
            if z not in dic:
                dic[z] = 0
            dic[z] += 1

    T = []
    for k, v in dic.items():
        if v > 2:
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
    
    
#def cnn_feature(flag):
#    if flag == 'train':
#        df = pd.read_csv(train_path, sep='\t', header=None, encoding='utf8')
#    else:
#        df = pd.read_csv(test_path, sep='\t', header=None, encoding='utf8')
#    
#    dic, max_id = read_dic()
#    
#    T = []
#    X = df.values
#    for i in range(len(X)):
#        if i%1000 == 0:
#            print (i)
##        if i == 2000:
##            break
#            
#        x = X[i]
#
#        t = [x[0], -1]
#        if flag == 'train':
#            t = [x[0], x[2]]
#       
##        if len(x[1]) > 10000:
##            x[1] = x[1][-10000:]
#        if len(x[1]) > max_seq_len:
#            x[1] = x[1][:max_seq_len]
#
#        sn = []
#        for z in x[1]:
#            if z in dic:
#                sn.append(dic[z])
#            else:
#                sn.append(max_id)
#        t = t + sn
#        T.append(t)
#    df = pd.DataFrame(T)
#    #df = df.astype('int')
#    df.to_csv('feature/'+flag+'.tsv', index=False, header=None, encoding='utf8')
    


    
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
        zz = x[1][:]
        t = [x[0], -1]
        if flag == 'train':
            t = [x[0], x[3]]
       
        #if len(x[1]) > 10000:
        if len(zz) > max_seq_len:
            zz = zz[:max_seq_len]

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

import codecs
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
        df = pd.DataFrame(X, columns=df.columns)
        print (files1[i])
        df.to_csv(files1[i], sep='\t', index=False, header=False, encoding='utf8')
        

def run(nrows=None):
    replace_train_test(nrows)  #step1
    
    get_word_id() #step2
    
    cnn_feature('train', nrows) #step3
    cnn_feature('test', nrows) #step4     
        
        
def main():
    #replace_train_test()
    
    get_word_id()
    
    cnn_feature('train')
    cnn_feature('test')
    
    #dense_feat()
    #find_zuiming()
    #read_zui(30)
#    zuiming_feat()
    
    #print (get_onehot(10000000))
    #string = '公诉50元机关梅510,000余元州市梅52万余元江区人4，999余元民检察院'
    #string = replace_money(string)
    #print(string)
    
    #t = '公诉机关梅州市梅江区人民检察院。被告人钟某。辩护人吴亦辉，系广东法泰律师事务所律师。梅州市梅江区人民检察院以区检诉刑诉（2014）257号起诉书指控被告人钟某犯信用卡诈骗罪，于2014年10月30日向本院提起公诉。送达起诉书副本及开庭审理时，被告人钟某自愿认罪，同意本案适用简易程序审理。本院适用简易程序，依法组成合议庭，公开开庭审理了本案。梅州市梅江区人民检察院指派代理检察员李丽玲出庭支持公诉，被告人钟某及其辩护人吴亦辉到庭参加了诉讼。现已审理终结。被告人钟某对公诉机关指控的犯罪事实、罪名无异议，请求从轻处罚。辩护人提出公诉机关指控的信用卡诈骗数额中有利息和滞纳金，应予以剔除，并提出公诉机关起诉书指控的第2、3项的犯罪事实，系被告人在司法机关未掌握的情况下主动交代的同种罪行和同种较重罪行，可酌情从轻和应当从轻处罚的。1、2012年2月，被告人钟某在中国建设银行梅州分行申请办理了卡号为5324582022516053（2013年1月更换为5324582023332427）的建设银行龙卡信用卡后透支用于赌博，经发卡银行多次催收均未还款，截止2014年7月21日欠本金人民币29681.07元、欠利息和滞纳金共人民币11753.27元。2、2012年8月，被告人钟某在广发银行梅州分行申请办理了卡号为6225581180004264的广发银行白金信用卡后透支用于赌博，经发卡银行多次催收均未还款，截止2014年8月14日欠本金人民币62393.71元、利息和滞纳金共人民币23675.95元。3、2012年3月，被告人钟某在中国工商银行梅州分行申请办理了卡号为6222300225966624的中国工商银行牡丹贷记信用卡后透支用于赌博，经发卡银行多次催收均未还款，截止2014年8月18日欠本金人民币9909.57元、利息和滞纳金共人民币6761.55元。被告人钟某使用以上三张信用卡透支，欠银行本金人民币101984.35元、欠利息和滞纳金共人民币42190.77元。被告人钟某归案后，其家属向中国建设银行梅州分行退赔了人民币40000元。上述事实，被告人钟某在开庭审理过程中亦无异议，且有书证被害单位银行的报案材料、申请、交易记录、账户追讨情况表，被告人钟某被抓获经过证明，身份证明，被害人被害单位负责人陈某新、潘某球、黄某政的陈述，退赔款项的情况证明，被告人钟某的供述等证据证实，足以认定。'
    #x = extract_amount_involved(t)
    #print (x)
    
if __name__ == '__main__':
    main()