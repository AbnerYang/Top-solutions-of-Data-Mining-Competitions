# 源码文档目录结构说明<br>

### data:存放原始数据<br>
----train.txt：训练集原始数据集（比赛方提供）<br>
----test.txt：测试集原始数据集（比赛方提供）<br>
----stop.txt：停用词信息（选手提交文档中包含）<br>
----map.txt：地域信息编码表（选手提交文档中包含）<br>

### feature:存放各选手自处理后的数据<br>
----lzp：存放user1处理后的数据<br>
----ljh：存放user2处理后的数据<br>
----yyt：存放user3处理后的数据<br>

### model：存放各自选手run.py产生的model<br>
----lzp：存放user1的model<br>
------money:罚金模型<br>
------laws：法文模型<br>
----ljh：存放user2的model<br>
------money:罚金模型<br>
------laws：法文模型<br>
----yyt：存放user3的model<br>
------money:罚金模型<br>
------laws：法文模型<br>

### result：存放各自选手run.py产生的result<br>
----lzp：存放user1的result<br>
------money:罚金<br>
-------- fajin.zi.jsonprob.tsv 罚金字级别预测中间结果<br>
-------- fajin.ci.jsonprob.tsv 罚金词级别预测中间结果<br>
------laws：法文<br>
-------- fawen.zi.jsonprob.tsv 罚金字级别预测中间结果<br>
-------- fawen.ci.jsonprob.tsv 罚金词级别预测中间结果<br>
----ljh：存放user2的result<br>
------money:罚金<br>
--------jh_penalty_cnn_blending_prob.csv 罚金CNN预测中间结果<br>
--------jh_penalty_dense_cnn_blending_prob.csv 罚金Wide&CNN预测中间结果<br>
------laws：法文<br>
--------jh_laws_cnn_blending_prob.csv 法文CNN预测中间结果<br>
--------jh_laws_wide&cnn_blending_prob.csv 法文Wide&CNN预测中间结果<br>
----yyt：存放user3的result
------money:罚金<br>
--------textCNN(SoftMax)_all_prob_blend.csv 罚金1预测中间结果<br>
--------textCNN(SigMoid)_all_prob_blend.csv 罚金2预测中间结果<br>
--------textCNN(SoftMax)_9_prob_blend.csv 罚金3预测中间结果<br>
------laws：法文
--------textCNN_laws_all_prob_blend.csv 法文1预测中间结果<br>
-----final.fajin.tsv 融合后罚金中间结果<br>
-----final.fawen.tsv 融合后法文中间结果<br>
-----final.all.json 融合最终结果<br>

### code：存放各自的源码<br>
----lzp：存放user1的源码<br>
------run.py：执行user1源码<br>
----ljh：存放user2的源码<br>
------run.py：执行user2源码<br>
----yyt：存放user3的源码<br>
------money:罚金模型源码<br>
------laws：法文模型源码<br>
------preprocess.py：数据预处理源码<br>
------run.py执行user3源码<br>
----blend.py执行融合各个用户结果复现源码<br>

### 复现源码文档执行源码说明<br>
1.	执行各选手下源码目录下的run.py获得各个选手的模型和结果<br>
2.	执行src目录下的blend.py获得融合结果<br>
3.	融合结果的最终文件命名为：/result/final.all.json <br>



### 复现源码执行环境（主要Package）说明<br>
1.	Ubuntu14.04+Python2.7<br>
2.	Keras 2.0.9<br>
3.	tensorflow 1.4.0<br>
4.	numpy 1.13.3<br>
5.	pandas 0.20.3<br>
6.	h5py 2.2.1<br>
7.	jieba 0.38<br>
8.	tqdm 4.19.4<br>
9.	genism 0.13.3<br>


