# Brief Introduction <br>

bayes_candidateDict.py  利用朴素贝叶斯生成每条记录的候选店铺概率字典<br>
unstack_candidateDict.py 将每条记录的候选店铺概率字典展开成堆叠方式的dataframe，方便通过dataframe的排序操作，构造候选集<br>
basic_feature.py 用于构造最基本的特征<br>
knn_feature.py 用于构造经纬度knn特征<br>
model.py 重新封装了XGBoost，方便使用<br>
multiclass_differParamter.py 通过使用不同参数生成多份多分类概率结果，用于融合到二分类中<br>
binaryXGB.py 用于进行最终的二分类<br>
<br>
ps:由于比赛总结中主要是以复赛进行总结，因此初赛的代码和比赛总结中有出入，有兴趣的可以按比赛总结优化这份初赛的代码。<br>
