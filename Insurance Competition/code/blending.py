# -*- enconding:utf-8 -*-
import numpy as np 
import pandas as pd 
from frame import *
from model import *
from feature import *
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
import copy
import os
import sys
import path
import re
import time as T
sys.setdefaultencoding('utf-8')
reg = re.compile('.*?-([\d\.]+)\(.*?\)\(([PL]+)')

res_dirs = ['../result/']
old = False
test = False
if __name__ == '__main__':
    if not os.path.exists('../final'):
        os.makedirs('../final')
    if not old and not test:
        iddf = pd.read_csv('../data/df_id_test.csv', header=None)
        iddf.columns = ['id']
        WholeProbDf, WholeVoteDf = iddf.copy(), iddf.copy()
        ConcatDf = iddf.copy()
        each_folder_PFscore, each_folder_LFscore = [], []
        ProbNs, VoteNs = [], []
        for k, res_dir in enumerate(res_dirs):
            print('processing folder [%s]' % res_dir)
            files = os.listdir(res_dir)
            PFs, LFs, Pls, Lls = [], [], [], []
            for name in files:
                tup = reg.findall(name)[0]
                if tup[1] == 'P':
                    PFs.append(float(tup[0]))
                    Pls.append(name)
                elif tup[1] == 'L':
                    LFs.append(float(tup[0]))
                    Lls.append(name)
            each_folder_PFscore.append(np.mean(PFs))
            each_folder_LFscore.append(np.mean(LFs))
            #load all prob files
            probDf = iddf.copy()
            for i, name in enumerate(Pls):
                tmpdf = pd.read_csv(res_dir + name, header=None)
                tmpdf.columns = ['id', 'prob' + str(i)]
                probDf = probDf.merge(tmpdf, on='id', how='left')
            probMat = probDf.drop(['id'], axis=1).values
            ConcatDf = pd.merge(ConcatDf, probDf, on='id', how='left')
            PFVec = np.array(PFs).reshape(-1, 1)
            ids = probDf.id.values
            vals = np.matmul(probMat, PFVec) / np.sum(PFVec)
            BlendProb = pd.DataFrame(data=[[ids[i], vals[i][0]] for i in range(len(vals))], index=probDf.index, columns=['id', 'prob' + str(k)])
            ProbNs.append('prob' + str(k))
            WholeProbDf = WholeProbDf.merge(BlendProb, on='id', how='left')
            #load all label files
            voteDf = iddf.copy()
            Ns = []
            for i, name in enumerate(Lls):
                tmpdf = pd.read_csv(res_dir + name, header=None)
                tmpdf.columns = ['id', 'vote' + str(i)]
                Ns.append('vote' + str(i))
                voteDf = voteDf.merge(tmpdf, on='id', how='left')
                
            tmpS = voteDf[Ns].mean(axis=1)
            tmpS.name = 'vote' + str(k)
            VoteNs.append('vote' + str(k))
            voteDf = pd.concat([voteDf.id, tmpS], axis=1)
            
            WholeVoteDf = WholeVoteDf.merge(voteDf, on='id', how='left')
        ConcatDf.to_csv('../concatdf.csv', header=False, index=False)
        probMat = WholeProbDf.drop(['id'], axis=1).values
        PFVec = np.array(each_folder_PFscore).reshape(-1, 1)
        ids = WholeProbDf.id.values
        vals = np.matmul(probMat, PFVec) / np.sum(PFVec)
        FinalBlendProb = pd.DataFrame(data=[[ids[i], vals[i][0]] for i in range(len(vals))] , 
                                        index=WholeProbDf.index, columns=['id', 'BlendProb'])
        
        voteMat = WholeVoteDf.drop(['id'], axis=1).values
        LFVec = np.array(each_folder_LFscore).reshape(-1, 1)
        ids = WholeVoteDf.id.values
        vals = np.matmul(voteMat, LFVec) / np.sum(LFVec)
        FinalBlendVote = pd.DataFrame(data=[[ids[i], vals[i][0]] for i in range(len(vals))], 
                                        index=WholeVoteDf.index, columns=['id', 'BlendVote'])
        
        
        curr = T.strftime('%d-%H-%M', T.localtime())
        store(FinalBlendProb.BlendProb, FinalBlendProb[['id']], curr + 'final_resfile(from prob)', 'final')
        store(FinalBlendVote.BlendVote, FinalBlendVote[['id']], curr + 'final_resfile(from vote)', 'final')
    if old and not test:
        print('constructing')
    if test:
        nsample = 4000
        each_folder_PFscore, each_folder_LFscore = [], []
        ProbNs, VoteNs = [], []
        WholeRL = np.zeros((nsample, 1))
        FolderRL = np.zeros((nsample, len(res_dirs)))
        for k, res_dir in enumerate(res_dirs):
            print('processing folder [%s]' % res_dir)
            files = os.listdir(res_dir)
            PFs, LFs, Pls, Lls = [], [], [], []
            for name in files:
                tup = reg.findall(name)[0]
                if tup[1] == 'P':
                    PFs.append(float(tup[0]))
                    Pls.append(name)
                elif tup[1] == 'L':
                    LFs.append(float(tup[0]))
                    Lls.append(name)
            each_folder_PFscore.append(np.mean(PFs))
            each_folder_LFscore.append(np.mean(LFs))
            #load all prob files
            FileRL = np.zeros((nsample, len(Pls)))
            for i, name in enumerate(Pls):
                tmpdf = pd.read_csv(res_dir + name, header=None)
                tmpdf.columns = ['id', 'prob']
                FileRL[:, i] = tmpdf.prob.sort_values(ascending=False).values
            
            probMat = FileRL
            PFVec = np.array(PFs).reshape(-1, 1)
            vals = np.matmul(probMat, PFVec) / np.sum(PFVec)
            
            FolderRL[:, k] = vals[:, 0]
        
        probMat = FolderRL
        PFVec = np.array(each_folder_PFscore).reshape(-1, 1)
        WholeRL = np.matmul(probMat, PFVec) / np.sum(PFVec)
        df = pd.DataFrame(WholeRL)
        df.loc[df.iloc[:, 0] > 0.1, 0].hist(bins=40, color='c')
        plt.show()
        print('###################################')
        for sep in range(190, 261, 10):
            print('Top %d probability %.5f' % (sep, WholeRL[sep][0]))
        print('\n###################################')
        for prob in np.arange(0.170, 0.221, 0.005):
            print(u'Amount over [probability %.5f]: %d' % (prob, np.sum(WholeRL > prob)))
        df.to_csv('../meanTop.csv', header=False, index=False)
    

