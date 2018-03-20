# 简要说明
#### 硬件设备
内存: 128G
CPU: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz

#### 运行环境
Ubuntu 16.04.2 LTS 
Python 2.7.13

#### Python Package
xgboost
lightgbm
pandas
numpy

#### 文档目录结构<br>

##### data:存放原始数据文件夹（执行前需建好）<br>
------user_action_train.txt(主办方给出的文件)<br>
------user_action_test_items.txt(主办方给出的文件)<br>
------goods_train.txt(主办方给出的文件)<br>

##### features:存放构造的特征文件夹（要执行程序之前建好）<br>
------base文件夹（要执行程序之前建好）<br>
------localtrain文件夹（要执行程序之前建好）<br>
------onlinetrain文件夹（要执行程序之前建好）<br>

##### result:存放预测结果文件夹（要执行程序之前建好）<br>
------blend（要执行程序之前建好）<br>
------local-online-xgb（要执行程序之前建好）<br>
------local-online-lgb（要执行程序之前建好）<br>
------online-online-xgb（要执行程序之前建好）<br>
------online-online-lgb（要执行程序之前建好）<br>

##### src（要执行程序之前建好）<br>
---model.py  封装相关的xgb 模型和Lgb模型<br>
---feature.py 构造模型所用的训练集和测试集（特征工程）<br>
---frame.py 封装相关的训练 预测 cv的结构<br>
---main.py 执行函数 ，执行该文件 得到预测结果。<br>

##### imp存放特征重要性文件（要执行程序之前建好）<br>


