# ReadMe

#### 1.修改featuring.py文件的路径<br>

PHASE1_TRAIN_PATH = '../dataSets/training/'<br>
PHASE1_TEST_PATH =  '../dataSets/testing_phase1/'<br>
PHASE2_PATH = '../dataSet_phase2/'<br>
FEATURES_PATH = '../features/'<br>

其中，<br>
PHASE1_TRAIN_PATH 是初赛的train数据的路径<br>
PHASE1_TRAIN_PATH 是初赛的test数据的路径<br>
PHASE2_PATH 是决赛数据的路径<br>
FEATURES_PATH 是生成特征的保存路径，与ym做bagging所用特征路径对应<br>

#### 2.修改xgb-training.py文件的路径<br>
FEATURES_PATH = '../features/'<br>
RESULT_PATH =  '../result/'<br>
RESULT_NAME = 'xgboost'<br>

其中，<br>
FEATURES_PATH 是featuring.py生成特征的保存路径，与ym做bagging所用特征路径对应<br>
RESULT_PATH 是输出结果的路径<br>
RESULT_NAME 是输出文件名<br>

文件执行顺序<br>
python featuring.py<br>
python xgb-training.py<br>
