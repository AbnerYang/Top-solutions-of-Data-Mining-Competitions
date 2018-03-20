path = "E:/competition_2016/MagicMirror/"
setwd(path)
#---ensemble linear
ensemble_linear_Result1 = read.csv("3wResult/ensemble.linear.2w.7631.csv",header = TRUE, stringsAsFactors = FALSE)
#---ensemble linear
ensemble_linear_Result2 = read.csv("3wResult/ensemble.linear.2w.7670.csv",header = TRUE, stringsAsFactors = FALSE)
#---ensemble linear
ensemble_linear_Result3 = read.csv("3wResult/ensemble.linear.2w.7680.csv",header = TRUE, stringsAsFactors = FALSE)
#---lasso
lasso_Result = read.csv("3wResult/lasso1.2w.pred.csv",header = TRUE, stringsAsFactors = FALSE)
#---lr
lr_Result = read.csv("3wResult/lr.csv",header = TRUE, stringsAsFactors = FALSE)
#---xgb1
xgb1_Result = read.csv("3wResult/xgb.4-16.1691.round1550.eta0.02.train3w.test2w.csv",header = TRUE, stringsAsFactors = FALSE)
xgb11_Result = read.csv("3wResult/xgb.4-17.3952.round1350.eta0.02.train3w.test2w.csv",header = TRUE, stringsAsFactors = FALSE)

#---xgb2
xgb2_Result = read.csv("3wResult/xgb1750_feature_merge2.csv",header = TRUE, stringsAsFactors = FALSE)
#---xgb3
xgb3_Result = read.csv("3wResult/xgb_online_2.csv",header = TRUE, stringsAsFactors = FALSE)

testIdx = xgb1_Result[,1]
#---输出读取数据
print(dim(ensemble_linear_Result1))
print(dim(ensemble_linear_Result2))
print(dim(lasso_Result))
print(dim(lr_Result))
print(dim(xgb1_Result))
print(dim(xgb2_Result))
print(dim(xgb3_Result))

#--设定融合的数据
data = data.frame(xgb11_Result[,2], xgb2_Result[,2], xgb3_Result[,2], ensemble_linear_Result3[,2])
#--设定融合的权重
weight = c(1,1, 1,3)
#--融合函数
getBlend <- function(data, weight){
  result = rep(0, nrow(data))
  for(i in 1:ncol(data)){
    data[,i] = (data[,i]-min(data[,i]))/(max(data[,i]) - min(data[,i]))
    result = result + data[,i]*weight[i]
  }
  return(result)
}
#--得到融合结果
score = getBlend(data,weight)

#--online AUC Test
daily = read.csv("first round test data/daily_test.csv", header = TRUE, stringsAsFactors = FALSE)
final = read.csv("first round test data/final_test.csv", header = TRUE, stringsAsFactors = FALSE)

findOnlineTestResult(ensemble_linear_Result1[,2], testIdx, daily, final) 
findOnlineTestResult(ensemble_linear_Result2[,2], testIdx, daily, final) 
findOnlineTestResult(ensemble_linear_Result3[,2], testIdx, daily, final) 
findOnlineTestResult(lasso_Result[,2], testIdx, daily, final) 
findOnlineTestResult(lr_Result[,2], testIdx, daily, final) 
findOnlineTestResult(xgb11_Result[,2], testIdx, daily, final) 
findOnlineTestResult(xgb2_Result[,2], testIdx, daily, final) 
findOnlineTestResult(xgb3_Result[,2], testIdx, daily, final) 

findOnlineTestResult(score, testIdx, daily, final) 

#--结果相关性对比
#--xgb1与xgb2
cor(xgb1_Result[,2], xgb2_Result[,2])
#--xgb1与lasso
cor(xgb1_Result[,2], xgb3_Result[,2])
#--xgb2与lasso
cor(xgb2_Result[,2], xgb3_Result[,2])

#--ensemble_linear_Result1与ensemble_linear_Result2
cor(ensemble_linear_Result1[,2], ensemble_linear_Result2[,2])
#--ensemble_linear_Result1与xgb1
cor(ensemble_linear_Result1[,2], xgb1_Result[,2])
#--ensemble_linear_Result1与xgb2
cor(ensemble_linear_Result1[,2], xgb2_Result[,2])
#--ensemble_linear_Result1与lasso
cor(ensemble_linear_Result1[,2], lasso_Result[,2])
#--ensemble_linear_Result1与result
cor(ensemble_linear_Result1[,2], score)


#--xgb_1与result
cor(xgb1_Result[,2], score)
#--xgb_2与result
cor(xgb2_Result[,2], score)
#--lasso与result
cor(lasso_Result[,2], score)