
import pandas as pd
import numpy as np


def save_fajin(ids, preds, outfile):
    res_path = outfile 
    sw = open(res_path, 'w')
    T= []
    for i in range(len(ids)):
        T.append([ids[i]] + list(preds[i]))
        t = list(preds[i])
        t = t.index(max(t)) + 1
        #{"id": "32", "penalty": 1, "laws": [1, 2, 3, 4]}
        sw.write('{"id": "'+str(ids[i])+'", "penalty": '+str(t)+', "laws": []}' + '\n')
    sw.close()
    df = pd.DataFrame(T)
    df.to_csv(res_path+'prob.tsv', index=False, header=False, float_format = '%.8f')
    

def save_fawen(ids, preds, outfile):
    res_path = outfile
    prob_path = res_path + '.prob.tsv'
    X = []
    T = []
    for i in range(len(ids)):
        x = [ids[i]]
        t = []
        
        T.append(x + list(preds[i]))
        
        for j in range(len(preds[i])):
            if preds[i][j] > 0.5:
                t.append(str(j+1))
        x.append(','.join(t))
        X.append(x)
        
    df = pd.DataFrame(X)
    df.to_csv(res_path, index=False, header=False)
    
    df = pd.DataFrame(T)
    df.to_csv(prob_path, index=False, header=False, float_format = '%.6f')


    
def ensemble_fajin(files, weis, outfile):
    print (np.array(weis).sum())
    
    res = []
    for i in range(len(files)):
        df = pd.read_csv(files[i], header=None)
        ids = df[0].values
        df.drop([0], axis=1, inplace=True)
        res.append(df.values)
        
    preds = []
    for i in range(len(res[0])):
        for j in range(len(files)):
            res[j][i] = res[j][i]/res[j][i].sum()
            
            if j == 0:
                t = res[j][i]*weis[j]
            else:
                t += res[j][i]*weis[j]
        preds.append(t)
        
    save_fajin(ids, preds, outfile)
    
    
    
    
def ensemble_fawen(files, weis, outfile):
    t = np.array(weis).sum()
    print (t)
    if t != 1.0:
        print ('....')
        return 
    
    ids = []
    P = []
    for i in range(len(files)):
        df = pd.read_csv(files[i], header=None) #, nrows=100
        ids = df[0].values
        titles = df.columns
        df = df[titles[1:]]
        p = df.values
        P.append(p)


    res = []
    for i in range(len(P[0])):
        r = []
        for j in range(len(P)):
            if j == 0:
                r = P[0][i]*weis[j]
            else:
                r = r + P[j][i]*weis[j]
        #r = r/len(P)
        res.append(r)
    #print (res[0])
    save_fawen(ids, res, outfile)
    
    
def ensemble_all(in1, in2, out):
    df1 = pd.read_csv(in1, header=None)
    df2 = pd.read_csv(in2, header=None)
    
    X1 = df1.values
    X2 = df2.values
    
    sw = open(out, 'w')
    
    for i in range(len(X1)):
        t = list(X1[i][1:])
        t = t.index(max(t)) + 1
        #print ((X1[i][0]), t, X2[i][1])
        if str(X2[i][1]) == 'nan':
            X2[i][1] = ''
            
        sw.write('{"id": "'+str(int(X1[i][0]))+'", "penalty": '+str(t)+', "laws": ['+X2[i][1]+']}' + '\n')
        
        #sw.write('{"id": "'+str(int(X1[i][0]))+'", "penalty": '+str(8)+', "laws": ['+X2[i][1]+']}' + '\n')

    sw.close()    



#----------------------------money--------------------------------------
#--lzp
fajin_lzp_zi = '../result/lzp/money/fajin.zi.jsonprob.tsv'
fajin_lzp_ci = '../result/lzp/money/fajin.ci.jsonprob.tsv'
#--ljh
fajin_ljh_cnn = '../result/ljh/money/jh_penalty_cnn_blending_prob.csv'
fajin_ljh_wcnn = '../result/ljh/money/jh_penalty_dense_cnn_blending_prob.csv'
#--yyt
fajin_yyt_sof_all = '../result/yyt/money/textCNN(SoftMax)_all_prob_blend.csv'
fajin_yyt_sig_all = '../result/yyt/money/textCNN(SigMoid)_all_prob_blend.csv'
fajin_yyt_sof_9 = '../result/yyt/money/textCNN(SoftMax)_9_prob_blend.csv'      

#----------------------------laws--------------------------------------
#--lzp
fawen_lzp_zi = '../result/lzp/laws/fawen.zi.tsv.prob.tsv'
fawen_lzp_ci = '../result/lzp/laws/fawen.ci.tsv.prob.tsv'
#--ljh
fawen_ljh_cnn = '../result/ljh/laws/jh_laws_cnn_blending_prob.csv'
fawen_ljh_wcnn = '../result/ljh/laws/jh_laws_wide&cnn_blending_prob.csv'
#--yyt
fawen_yyt = '../result/yyt/laws/textCNN_laws_all_prob_blend.csv'

    
final_fajin = '../result/final.fajin.tsv'
final_fawen = '../result/final.fawen.tsv'

final_all = '../result/final.all.json'


if __name__ == '__main__':
    #融合罚金
    files = [fajin_lzp_zi, fajin_lzp_ci, fawen_ljh_wcnn, fajin_ljh_cnn, fajin_yyt_sof_all, fajin_yyt_sof_9, fajin_yyt_sig_all]
    weis = [0.2, 0.5, 0.09, 0.05, 0.09, 0.05, 0.02]
    ensemble_fajin(files, weis, final_fajin)
    
    
    #融合法文
    files = [fawen_lzp_zi, fawen_lzp_ci, fawen_ljh_wcnn, fawen_ljh_cnn, fawen_yyt]
    weis = [0.25, 0.55, 0.1, 0.05, 0.05]
    ensemble_fawen(files, weis, final_fawen)
    

    #合并罚金和法文
    ensemble_all(final_fajin+'prob.tsv', final_fawen, final_all)
    
    