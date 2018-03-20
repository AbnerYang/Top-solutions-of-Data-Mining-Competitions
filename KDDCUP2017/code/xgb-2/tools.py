#coding:utf-8

import pandas as pd
import numpy as np


#保存特征重要性
def feature_important(bst, filepath):
    feature_score = bst.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    x1 = []
    x2 = []
    for (key,value) in feature_score:
        x1.append(key)
        x2.append(value)
    feat_im = pd.DataFrame({"feature_name":x1, "score":x2})
    feat_im.to_csv(filepath, index=False)
    
#删除重复的列    
def remove_dupl_col(df):
    # remove duplicated columns
    remove = []
    c = df.columns
    for i in range(len(c)-1):
        #print "col:", i
        v = df[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,df[c[j]].values):
                remove.append(c[j])
    df.drop(remove, axis=1, inplace=True)    
    print remove
    return df, remove

#删除只有一个值的列
def remove_cons_col(df):
    # remove constant columns
    remove = []
    for col in df.columns:
        if np.array(df[col]).std() == 0:  #  < 0.1:
            remove.append(col)
    df.drop(remove, axis=1, inplace=True)
    print remove
    return df, remove