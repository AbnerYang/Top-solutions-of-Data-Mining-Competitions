
import numpy as np
import pandas as pd


#加载除第三方特征以为的所有做好的特征
def loadData(path):
    train_x = pd.read_csv(path, sep=',')
    
    uid = train_x[['V1']]
    
    train_x.drop(['V1'], axis=1, inplace=True)
    train_y = pd.read_csv("data/idx.target.csv", sep=',')
    train_y.drop(['Idx'], axis=1, inplace=True)
    
    return train_x, train_y, uid
    
    
    
#第三方两列的相除的定义，针对不可除的不同情况定义
def arr_divide(x1, x2):
    x = []
    for k in xrange(0, len(x1)):
        
        if x1[k] == -1 and x2[k]== -1:
            x.append(-8)
        elif x1[k] == -1 and x2[k] == 0:
            x.append(-7)
        elif x1[k] == 0 and x2[k] == 0:
            x.append(-6)
        elif x1[k] == 0 and x2[k] == -1:
            x.append(-5)
        elif x1[k] > 0 and x2[k] == -1:
            x.append(-4)
        elif x1[k] == 0 and x2[k] > 0:
            x.append(-3)
        elif x1[k] == -1 and x2[k] > 0:
            x.append(-2)
        else:
            x.append(x2[k]*1.0/x1[k])
                   
    return x

#第三方统计特征，包括平均、标准差、求和、最大、最小、大于平均的个数、缺失个数，以及缺失填充平均值
def third_statistical(line_arr, dfarr, prefix):
    df = pd.concat(line_arr, axis=1)
    df = df.replace(np.nan, -1)
    line = df.shape[0]
    avg = []
    std = []
    sum = []
    max = []
    min = []
    large_avg = []
    missing_count = []
    
    for i in xrange(0, line):
        arr = list(df.iloc[i,])
        
        b = 0
        t = 0
        cout = 0
        for x in arr:
            if x == -1:
                b += 1
            else:
                t += 1
                cout += x
        missing_count.append(b)        
              
        if t > 0:
            cout /= t
            for k in xrange(0, len(arr)):
                if arr[k] == -1:
                    arr[k] = cout
            
        np_arr = np.array(arr)
        
        tmp_avg = np_arr.mean()
        avg.append(tmp_avg)
        std.append(np_arr.std())
        sum.append(np_arr.sum())
        max.append(np_arr.max())
        min.append(np_arr.min())
        
        a = 0
        for x in arr:
            if x > tmp_avg:
                a += 1
        large_avg.append(a)

        
    tm_df = pd.DataFrame({prefix+'avg':avg})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'std':std})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'sum':sum})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'max':max})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'min':min})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'large_avg':large_avg})
    dfarr.append(tm_df)
    
    tm_df = pd.DataFrame({prefix+'missing_count':missing_count})
    dfarr.append(tm_df)   

#计算第三方数据的特征
#包括同比、环比以及统计特征
def third_party(path):
    df = pd.read_csv(path, sep=',')
    
    dfarr = []
    
    for i in xrange(1, 7):
        print  "1 i:", i
        for j in xrange(1, 18):
            df1 = df['ThirdParty_Info_Period' + str(i) +'_' + str(j)]
            df2 = df['ThirdParty_Info_Period' + str(i+1) +'_' + str(j)]
            x1 = df1.values
            x2 = df2.values
            
            x = arr_divide(x1, x2)    
   
            tm_df = pd.DataFrame({'ThirdParty_'+str(i+1)+'/'+str(i)+'_'+str(j):x})
            dfarr.append(tm_df)
    
      
    for i in xrange(1, 8):
        print  "2 i:", i
        for j in xrange(1, 17):
            for k in xrange(j+1, 18):
                df1 = df['ThirdParty_Info_Period' + str(i) +'_' + str(j)]
                df2 = df['ThirdParty_Info_Period' + str(i) +'_' + str(k)]
                x1 = df1.values
                x2 = df2.values
                
                x = arr_divide(x1, x2)    
       
                tm_df = pd.DataFrame({'ThirdParty_'+str(i)+'_'+str(k) + '/' +str(j):x})
                dfarr.append(tm_df)
               
                
   
    for i in xrange(1, 8):
       line_arr = []
       for j in xrange(1, 18):
           df1 = df['ThirdParty_Info_Period' + str(i) +'_' + str(j)]
           line_arr.append(df1)
       third_statistical(line_arr, dfarr, 'period_'+str(i)+'_deal_')
       
    for i in xrange(1, 18):
        line_arr = []
        for j in xrange(1, 8):
            df1 = df['ThirdParty_Info_Period' + str(j) +'_' + str(i)]
            line_arr.append(df1)
        third_statistical(line_arr, dfarr, 'period_deal_'+str(i)+'_')
    
    
    feat_df = pd.concat(dfarr, axis=1)    
    return feat_df
    
#删除标准差为0的列，删除相同值的列
def dropFeature(train, test):
    # remove constant columns
    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)    
    
    
    # remove duplicated columns
    remove = []
    c = train.columns

    dic = {}
    for i in xrange(0, len(c)):
        dic[c[i]] = train[c[i]].values.sum()
    
    for i in range(len(c)-1):
        print i
        v = train[c[i]].values
        for j in range(i+1,len(c)):
            if dic[c[i]] == dic[c[j]]:
                if np.array_equal(v,train[c[j]].values):
                    remove.append(c[j])
    
    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)    
    
    return train, test
   

def main():
    
    ###########test
    test_x, test_y, uid_test = loadData("data/feature_3422/test.x.select.csv")
    
    #测试1万的master
    test_third = third_party("data/Kesci_Master_9w_gbk_1_test_set.csv")
    

    
    test_x = pd.concat([test_x, test_third], axis=1)
    
    
    
    #########train
    train_x, train_y, uid_train = loadData("data/feature_3422/train.x.select.csv")
    
    #训练8万的master
    train_third = third_party("data/master_3w_2w_3w.csv")
    
    train_x = pd.concat([train_x, train_third], axis=1)

    ###删除无用的特征
    train_x, test_x = dropFeature(train_x, test_x)
    
    print train_x.shape[0], test_x.shape[0]
    print train_x.shape[1], test_x.shape[1]
    

    print test_x
    train = pd.concat([uid_train, train_x], axis=1)
    test = pd.concat([uid_test, test_x], axis=1)
    
    #输出特征
    train.to_csv("data/feature_3952/train.3952.csv", index=False)
    test.to_csv("data/feature_3952/test.3952.csv", index=False)
    




    
if __name__ == "__main__":
    main()
    
    
    
    
    