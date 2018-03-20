path = "E:/competition_2016/MagicMirror"
setwd(path)
workPath = getwd()
logPath = "log/cv.txt"
#--Read Feature
train.x.8w = read.csv("feature_8w/train.3952.csv",stringsAsFactors = FALSE, header = TRUE)
test.x.8w = read.csv("feature_8w/test.3952.csv", stringsAsFactors = FALSE, header = TRUE)
#--Read Target
target = read.csv("feature/target.csv", header = TRUE, stringsAsFactors = FALSE)
xgb_4_13_Result = read.csv("8wResult/xgb.4-13.csv",header = TRUE, stringsAsFactors = FALSE)
target = target$target
testIdx = xgb_4_13_Result$Idx
#---Xgb 8w cv Result
ModelList = c(1)
roundList = c(3000)
cvResult = CVFrameWork(train_x_select, target, ModelList, roundList, logPath, floor = 1)

#---Online 8w Result
predict = xgbOnlineTreeModel(train.x.8w, test.x.8w, target, iter = 1,se = 1, round = 2280)

result = data.frame("Idx" = testIdx, "score" = predict[,1])
write.csv(result,"8wResult/Xgb_online_final_1.csv", row.names = FALSE)