# -*-encoding:utf-8 -*-
import fajin_feat_zi, fajin_feat_ci
import fajin_dnn_zi, fajin_dnn_ci

import fawen_feat_zi, fawen_feat_ci
import fawen_dnn_zi, fawen_dnn_ci


def process():
    # nrows = 100 #test code 读取1000行进行代码测试
    nrows = None #run 使用全量数据
    
    #基于字的罚金
    fajin_feat_zi.run(nrows)
    fajin_dnn_zi.run(0) #training
    fajin_dnn_zi.run(1) #predict
    


    #基于词的罚金
    fajin_feat_ci.run(nrows)
    fajin_dnn_ci.run(0) #training
    fajin_dnn_ci.run(1) #predict
    


    #基于字的法文
    fawen_feat_zi.run(nrows)
    fawen_dnn_zi.run(0)
    fawen_dnn_zi.run(1)


    #基于词的法文
    fawen_feat_ci.run(nrows)
    fawen_dnn_ci.run(0)
    fawen_dnn_ci.run(1)
    
    
if __name__ == '__main__':
    process()