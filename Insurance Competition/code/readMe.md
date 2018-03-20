# Brief Introduction

feature.py 负责特征工程， 运行结果保存在../feature/下<br>
frame.py 负责提供一个抽象的算法调用接口<br>
model.py 包含各种算法<br>
nmf.py   提供非负矩阵分解算法<br>
main.py  主程序，负责使用不同的数据子集上训练模型，并预测结果，运行结果保存在../result/下<br>
blending.py  负责预测结果的融合，运行结果保存在../final/下<br>


* 使用方法<br>
(1) 将数据放在data文件夹下，并将其与code文件夹放在同一路径下<br>
(2) 将cmd工作路径切换到code文件夹内<br>
(2) 运行python main.py -h 可以查看帮助信息<br>
<br>
* 简单运行示例<br>
(1) python main.py<br>
(2) python blending.py<br>
<br>
ps:由于比赛总结中主要是以复赛进行总结，因此初赛的代码和比赛总结中有出入，有兴趣的可以按比赛总结优化这份初赛的代码。<br>
