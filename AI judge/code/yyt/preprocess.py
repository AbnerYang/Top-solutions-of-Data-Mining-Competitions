# -*- encoding:utf-8 -*-
from __future__ import division
from __future__ import absolute_import
import pandas as pd
import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
import jieba
import warnings
import codecs
import copy
import re
from tqdm import tqdm

def convertChineseDigitsToArabic (chinese_digits, encoding="utf-8"):
    chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
            u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
            u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
            u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
            u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
            u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4,
            u'亿':10 ** 8, u'億':10 ** 8, u'幺': 1,
            u'０':0, u'１':1, u'２':2, u'３':3, u'４':4,
            u'５':5, u'６':6, u'７':7, u'８':8, u'９':9}

    if isinstance (chinese_digits, str):
        chinese_digits = chinese_digits.decode (encoding)

    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result

def get_alpha(number, flag=False):
    if flag:
        number *= 10000
    

    list1 = [30, 100, 300, 1000, 2000, 3000, 4000, 5000, 7000, 10000, 20000, 50000, 100000, 500000]
    list2 = ['QA', 'QB', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QJ', 'QK', 'QL', 'QM', 'QN', 'QO','QP']

    i = 0
    while i<len(list1):
        if number<list1[i]:
            break
        
        i += 1       
    return list2[i]

def get_date(date, flag=False):
    year = int(date.split('年')[0])
    month = int(date.split('年')[1].split('月')[0])

    list11 = [1970, 1980, 1990, 1993, 1999, 2003, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    list21 = ['YA', 'YB', 'YC', 'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YJ', 'YK', 'YL', 'YM', 'YN', 'YO', 'YP', 'YQ','YR']
    
    list12 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    list22 = ['MA', 'MB', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI', 'MJ', 'MK', 'ML','MM']

    i = 0
    while i<len(list11):
        if year<list11[i]:
            break
        
        i += 1  
    
    j = 0
    while j<len(list12):
        if month<list12[j]:
            break
        
        j += 1
    return list21[i]+' '+list22[j]

def get_ke(ke, flag='KE'):
    ke = float(ke.replace(',','').replace('，',''))
    
    list11 = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    list21 = [flag+'A', flag+'B', flag+'C', flag+'D', flag+'E', flag+'F', flag+'G', flag+'H', flag+'I', flag+'J']
    

    i = 0
    while i<len(list11):
        if ke<list11[i]:
            break
        
        i += 1  

    return list21[i]

def docPreprocess(doc):
    doc = doc.decode('utf-8')
    doc = doc.replace(u'余元', u'元').replace(u'万余元', u'万元').replace(u'余万元', u'万元')
    
    r1 = re.compile(u'(\\d+((,\\d+)|(，\\d+))*(\.\\d+)?)元')
    numbers = r1.findall(doc)
    for number in numbers:
        number = number[0]
        k = float(number.replace(u',',u'').replace(u'，',u''))
        alpha = get_alpha(k)
        doc = doc.replace(number+u'元', alpha)
        
    r2 = re.compile(u'(\\d+((,\\d+)|(，\\d+))*(\.\\d+)?)万元')
    numbers = r2.findall(doc)
    for number in numbers:
        number = number[0]
        k = float(number.replace(u',',u'').replace(u'，',u''))
        alpha = get_alpha(k, True)
        doc = doc.replace(number+u'万元', alpha)
    
    r = re.compile(u'[一二三四五六七八九]+[零十百千万亿一二三四五六七八九]*元')
    numbers = r.findall(doc)
    for number in numbers:
        k = convertChineseDigitsToArabic(number[:-1])
        alpha = get_alpha(k)
        doc = doc.replace(number, alpha)
        
    return doc

def splitWord(query, stopwords):
    query = docPreprocess(query)
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        if word.rstrip() not in stopwords:
            word = word.replace(' ','')
            word = word.replace('"','')
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result.encode('utf-8')

def getDoc(data,stopwords):
    df = copy.deepcopy(data)
    doc = []
    for des in tqdm(df.description.values):
        res = splitWord(des, stopwords)
        doc.append(res)
    df['doc'] = doc

    return df

def run():
    stopwords = {}
    for line in codecs.open('../../data/stop.txt', 'r', 'utf-8'):
        stopwords[line.rstrip()] = 1

    train_raw = pd.read_table('../../data/train.txt',sep='\t',header=None)#,nrows = 200)
    test_raw = pd.read_table('../../data/test.txt',sep='\t',header=None)#, nrows = 200)

    train_raw.columns = ['ID','description','penalty','laws']
    test_raw.columns = ['ID','description']

    
    train = getDoc(train_raw,stopwords)
    test = getDoc(test_raw,stopwords)
    train.to_csv('../../feature/yyt/traindoc_money_law.csv', index = False)
    test.to_csv('../../feature/yyt/testdoc_money_law.csv', index = False)