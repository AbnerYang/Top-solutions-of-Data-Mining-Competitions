path = "E:/competition_2016/MagicMirror/"
setwd(path)
#---o
xgb_4_13_Result = read.csv("8wResult/xgb.4-13.csv",header = TRUE, stringsAsFactors = FALSE)
#---1
xgb1_Result = read.csv("8wResult/xgb.4-16.1691.round2800.eta0.02.train8w.test1w.csv",header = TRUE, stringsAsFactors = FALSE)
#---2
xgb2_Result = read.csv("8wResult/xgb2500_feature_2485.csv",header = TRUE, stringsAsFactors = FALSE)
#---4
lasso_Result = read.csv("8wResult/test.lasso.result7727.csv",header = TRUE, stringsAsFactors = FALSE)
#---3 
xgb3_Result = read.csv("8wResult/Xgb_online_final_1.csv",header = TRUE, stringsAsFactors = FALSE)

#---输出读取数据
print(dim(xgb_4_13_Result))
print(dim(xgb1_Result))
print(dim(xgb2_Result))
print(dim(lasso_Result))
#--设定融合的数据
data = data.frame(xgb1_Result[,2], xgb2_Result[,2], lasso_Result[,2])
#--设定融合的权重
weight = c(0.5,0.4, 0.1)
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
#--存储结果
result = data.frame("Idx" = xgb_4_13_Result[,1], "score" = score)
write.csv(result,file = paste(paste("result/4-16-",1,sep = ""),".csv",sep = ""), row.names = FALSE)

print(dim(result))

#--结果相关性对比
#--xgb1与xgb2
cor(xgb1_Result[,2], xgb2_Result[,2])
#--xgb1与lasso
cor(xgb1_Result[,2], lasso_Result[,2])
#--xgb2与lasso
cor(xgb2_Result[,2], lasso_Result[,2])

#--xgb_4-13与xgb1
cor(xgb_4_13_Result[,2], xgb1_Result[,2])
#--xgb_4-13与xgb2
cor(xgb_4_13_Result[,2], xgb2_Result[,2])
#--xgb_4-13与lasso
cor(xgb_4_13_Result[,2], lasso_Result[,2])
#--xgb_4-13与result
cor(xgb_4_13_Result[,2], result[,2])
#--xgb_1与result
cor(xgb1_Result[,2], result[,2])
#--xgb_2与result
cor(xgb2_Result[,2], result[,2])
#--lasso与result
cor(lasso_Result[,2], result[,2])

#--xgb_1与result
cor(xgb1_Result[,2], xgb3_Result[,2])
#--xgb_2与result
cor(xgb2_Result[,2], xgb3_Result[,2])
#--lasso与result
cor(lasso_Result[,2], xgb3_Result[,2])