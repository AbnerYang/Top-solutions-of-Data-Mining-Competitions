path = "E:/competition_2016/MagicMirror"
setwd(path)
workPath = getwd()
logPath = "log/cv.txt"
#--Read Feature
train.x.3w = read.csv("feature_3w/train.x.3w.4747.csv",stringsAsFactors = FALSE, header = TRUE)
test.x.3w = read.csv("feature_3w/test.x.3w.4747.csv", stringsAsFactors = FALSE, header = TRUE)
#--Read Target
round_1_train_master = read.csv("first round train data/PPD_Training_Master_GBK_3_1_Training_Set.csv", stringsAsFactors = FALSE, header = TRUE)
round_1_test_master = read.csv("first round test data/Kesci_Master_9w_gbk_2.csv", stringsAsFactors = FALSE, header = TRUE)
target = round_1_train_master$target
testIdx = round_1_test_master$Idx
#---Xgb 3w cv Result
ModelList = c(1)
roundList = c(1900)
cvResult = CVFrameWork(train.x.3w, target, ModelList, roundList, logPath, floor = 1)

#---Online 3w Result
predict = xgbOnlineTreeModel(train.x.3w, test.x.3w, target, iter = 1,se = 1, round = 1300)

result = data.frame("Idx" = testIdx, "score" = predict[,1])
write.csv(result,"3wResult/Xgb_online_2.csv", row.names = FALSE)

