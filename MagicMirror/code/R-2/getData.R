
path = "E:/competition_2016/MagicMirror"
setwd(path)

#==============================================处理级别 1======================================
#--获取Master数值型特征--
round_1_train_master = read.csv("first round train data/PPD_Training_Master_GBK_3_1_Training_Set.csv", stringsAsFactors = FALSE, header = TRUE)
round_1_test_master = read.csv("first round test data/Kesci_Master_9w_gbk_2.csv", stringsAsFactors = FALSE, header = TRUE)
round_2_train_master = read.csv("second round newly added data/Kesci_Master_9w_gbk_3_2.csv", stringsAsFactors = FALSE, header = TRUE)
round_2_test_master = read.csv("second round test data/Kesci_Master_9w_gbk_1_test_set.csv", stringsAsFactors = FALSE, header = TRUE)

train_master = rbind(round_1_train_master,round_1_test_master,round_2_train_master)
target = train_master$target
train_master_feature = train_master[,-(ncol(train_master)-1)] 
test_master_feature = round_2_test_master

testIdx = test_master_feature$Idx
ft = read.csv("featureType.csv", header = TRUE, stringsAsFactors = FALSE)

fn.cat=as.character(ft[ft[,2]=='Categorical',1])
fn.num=as.character(ft[ft[,2]=='Numerical',1])

#---未将NA填充-1的原始数据
train_master_num = train_master_feature[,fn.num]
test_master_num = test_master_feature[,fn.num]

train_master_cat = train_master_feature[,fn.cat]
test_master_cat = test_master_feature[,fn.cat]

a = findStrList(train_master_cat)

train_master_str = train_master_cat[,a]
test_master_str = test_master_cat[,a]

train_master_str = cleanCatTypeFeature(train_master_str)
test_master_str = cleanCatTypeFeature(test_master_str)

train_master_catTonum = train_master_cat[,-a]
test_master_catTonum = test_master_cat[,-a]
remove(a)

a = c(1,2,3,4,6,7,10)
train_master_catLoc = train_master_str[,a]
test_master_catLoc = test_master_str[,a]

train_master_catNoLoc = train_master_str[,-a]
test_master_catNoLoc = test_master_str[,-a]
remove(a)

#--Master 缺失值转为-1
train_master_num_NAToNeg1 = FillNAValue(train_master_num, c("","不详"), -1)
test_master_num_NAToNeg1 = FillNAValue(test_master_num, c("","不详"), -1)

train_master_catLoc_NAToNeg1 = FillNAValue(train_master_catLoc, c("","不详"), -1)
test_master_catLoc_NAToNeg1 = FillNAValue(test_master_catLoc, c("","不详"), -1)

train_master_catNoLoc_NAToNeg1 = FillNAValue(train_master_catNoLoc, c("","不详"), -1)
test_master_catNoLoc_NAToNeg1 = FillNAValue(test_master_catNoLoc, c("","不详"), -1)

train_master_catTonum_NAToNeg1 = FillNAValue(train_master_catTonum, c("","不详"), -1)
test_master_catTonum_NAToNeg1 = FillNAValue(test_master_catTonum, c("","不详"), -1)
#----存储
write.csv(train_master_num_NAToNeg1, "xgbfeature/train_master_num_NAToNeg1.csv", row.names = FALSE)
write.csv(train_master_catLoc_NAToNeg1, "xgbfeature/train_master_catLoc_NAToNeg1.csv", row.names = FALSE)
write.csv(train_master_catNoLoc_NAToNeg1, "xgbfeature/train_master_catNoLoc_NAToNeg1.csv", row.names = FALSE)
write.csv(train_master_catTonum_NAToNeg1, "xgbfeature/train_master_catTonum_NAToNeg1.csv", row.names = FALSE)

write.csv(test_master_num_NAToNeg1, "xgbfeature/test_master_num_NAToNeg1.csv", row.names = FALSE)
write.csv(test_master_catLoc_NAToNeg1, "xgbfeature/test_master_catLoc_NAToNeg1.csv", row.names = FALSE)
write.csv(test_master_catNoLoc_NAToNeg1, "xgbfeature/test_master_catNoLoc_NAToNeg1.csv", row.names = FALSE)
write.csv(test_master_catTonum_NAToNeg1, "xgbfeature/test_master_catTonum_NAToNeg1.csv", row.names = FALSE)


result = data.frame("Idx" = train_master_feature$Idx, "target" = target)
write.csv(result,"xgbfeature/target.csv", row.names = FALSE)


#====================================================处理级别2===========================================================


#--Master 连续性缺失值置为中位数  类别型置为众数
MissList = findMissPoint(train_master_num_NAToNeg1, test_master_num_NAToNeg1)
#MedianList = findMedianPoint(train_master_num_NAToNeg1, test_master_num_NAToNeg1, 0)
MeanList = findMeanPoint(train_master_num_NAToNeg1, test_master_num_NAToNeg1, 0)

train_master_num_fillNA = fillingMiss(train_master_num_NAToNeg1, MissList, MeanList, 0.2)
test_master_num_fillNA= fillingMiss(test_master_num_NAToNeg1, MissList, MeanList, 0.2)

#----类别型
MissList = findMissPoint(train_master_catLoc_NAToNeg1, test_master_catLoc_NAToNeg1)
ModeList = findModePoint(train_master_catLoc_NAToNeg1, test_master_catLoc_NAToNeg1, 0)

train_master_catLoc_fillNA = fillingMiss(train_master_catLoc_NAToNeg1, MissList, ModeList, 0.2)
test_master_catLoc_fillNA= fillingMiss(test_master_catLoc_NAToNeg1, MissList, ModeList, 0.2)

MissList = findMissPoint(train_master_catNoLoc_NAToNeg1, test_master_catNoLoc_NAToNeg1)
ModeList = findModePoint(train_master_catNoLoc_NAToNeg1, test_master_catNoLoc_NAToNeg1, 0)

train_master_catNoLoc_fillNA = fillingMiss(train_master_catNoLoc_NAToNeg1, MissList, ModeList, 0.2)
test_master_catNoLoc_fillNA= fillingMiss(test_master_catNoLoc_NAToNeg1, MissList, ModeList, 0.2)

#----
MissList = findMissPoint(train_master_catTonum_NAToNeg1, test_master_catTonum_NAToNeg1)
ModeList = findModePoint(train_master_catTonum_NAToNeg1, test_master_catTonum_NAToNeg1, 0)

train_master_catTonum_fillNA = fillingMiss(train_master_catTonum_NAToNeg1, MissList, ModeList, 0.2)
test_master_catTonum_fillNA= fillingMiss(test_master_catTonum_NAToNeg1, MissList, ModeList, 0.2)


#----存储二级处理后的特征

write.csv(train_master_num_fillNA, "xgbfeature/train_master_num_fillNA.csv", row.names = FALSE)
write.csv(train_master_catLoc_fillNA, "xgbfeature/train_master_catLoc_fillNA.csv", row.names = FALSE)
write.csv(train_master_catNoLoc_fillNA, "xgbfeature/train_master_catNoLoc_fillNA.csv", row.names = FALSE)
write.csv(train_master_catTonum_fillNA, "xgbfeature/train_master_catTonum_fillNA.csv", row.names = FALSE)

write.csv(test_master_num_fillNA, "xgbfeature/test_master_num_fillNA.csv", row.names = FALSE)
write.csv(test_master_catLoc_fillNA, "xgbfeature/test_master_catLoc_fillNA.csv", row.names = FALSE)
write.csv(test_master_catNoLoc_fillNA, "xgbfeature/test_master_catNoLoc_fillNA.csv", row.names = FALSE)
write.csv(test_master_catTonum_fillNA, "xgbfeature/test_master_catTonum_fillNA.csv", row.names = FALSE)


#====================================================第三级=================================================

name = names(train_master_num_NAToNeg1)
k = name[57:175]

train_master_thridParty = train_master_num_NAToNeg1[,k]
test_master_thridParty = test_master_num_NAToNeg1[,k]

train_master_thridParty_trend = getTrendFeature(train_master_thridParty)
test_master_thridParty_trend = getTrendFeature(test_master_thridParty)
write.csv(train_master_thridParty_trend, "xgbfeature/train_master_thridParty_trend.csv", row.names = FALSE)
write.csv(test_master_thridParty_trend, "xgbfeature/test_master_thridParty_trend.csv", row.names = FALSE)

#xgb 正置 train_master_num: 900 0.7441
#xgb 倒置 train_master_num: 2000 0.7449

#均值融合  0.1/0.9 = 0.748
#-- base train_master_num_fillNA 0.05  xgb 倒置 0.7446 2000
#-- base train_master_num_fillNA 0.2  xgb 倒置 0.746430 2000
#-- base train_master_num_fillNA 0.2  xgb 倒置 0.746208 2200
#-- base train_master_num_fillNA 0.4  xgb 倒置 0.746413 2200
#-- base train_master_num_fillNA 0.8  xgb 倒置 0.746061 2000

#-- base train_master_num_missSelect 0.8  xgb 倒置 0.7460 2000
#-- base train_master_num_missSelect 0.97  xgb 倒置 0.746117 2000
#-- base train_master_num_missSelect 0.98  xgb 倒置 0.7458 2000

#-- base train_x  xgb 倒置 0.7460 2200

#xgb 正置 train_master_num_fillNA 0.2 : 900 0.7448
#xgb 倒置 train_master_num_fillNA 0.2 : 2000 0.7464

#均值融合  0.1/0.9 = 0.749
#----存储二级处理后的特征

write.csv(train_master_num_fillNA, "feature/train_master_num_fillNA.csv", row.names = FALSE)
write.csv(train_master_num_missSelect, "feature/train_master_num_missSelect.csv", row.names = FALSE)
write.csv(train_master_num_missStat, "feature/train_master_num_missStat.csv", row.names = FALSE)

write.csv(test_master_num_fillNA, "feature/test_master_num_fillNA.csv", row.names = FALSE)
write.csv(test_master_num_missSelect, "feature/test_master_num_missSelect.csv", row.names = FALSE)
write.csv(test_master_num_missStat, "feature/test_master_num_missStat.csv", row.names = FALSE)

#=========================================处理级别2：one-hot编码==============================================
train_master_strcatLoc =  read.csv("feature/train_master_strcatLoc.csv",  header = TRUE, stringsAsFactors = FALSE)
train_master_strcatNoLoc =  read.csv("feature/train_master_strcatNoLoc.csv",  header = TRUE, stringsAsFactors = FALSE)

test_master_strcatLoc =  read.csv("feature/test_master_strcatLoc.csv",  header = TRUE, stringsAsFactors = FALSE)
test_master_strcatNoLoc =  read.csv("feature/test_master_strcatNoLoc.csv",  header = TRUE, stringsAsFactors = FALSE)

#--获取one-hot编码
strNoLoc = getOneHotFeature(train_master_strcatNoLoc, test_master_strcatNoLoc)

train_master_strcatNoLoc_oneHot = strNoLoc[[1]]
test_master_strcatNoLoc_oneHot = strNoLoc[[2]]
remove(strNoLoc)

strLoc = getOneHotFeature(train_master_strcatLoc[,-ncol(train_master_strcatLoc)], test_master_strcatNoLoc[,-ncol(train_master_strcatLoc)])

train_master_strcatLoc_oneHot = strLoc[[1]]
test_master_strcatLoc_oneHot = strLoc[[2]]
remove(strLoc)


#=========================================处理级别3：loginfo/userupdate特征==============================================

#--get Train and Test UserUpdate Feature

listUserUpdate = getUserUpdateFeature(trainMaster$Idx,testMaster$Idx,trainUserupdate,testUserupdate)

train.userUpdate.x = listUserUpdate[[1]]
test.userUpdate.x = listUserUpdate[[2]]

remove(listUserUpdate)
#====================================================================================

#--get Train and Test Loginfo Feature
list1 = getLoginfoFeature(trainMaster$Idx,testMaster$Idx,trainLogInfo,testLogInfo, 2)
list2 = getLoginfoFeature(trainMaster$Idx,testMaster$Idx,trainLogInfo,testLogInfo, 3)

train.loginfo.x = cbind(list1[[1]],list2[[1]])
test.loginfo.x = cbind(list1[[2]],list2[[2]])

remove(list1,list2)
#====================================================================================
#--get Action Num
actionNum.logInfo.train = getActionTimes(trainMaster$Idx,trainLogInfo)
actionNum.logInfo.test = getActionTimes(testMaster$Idx,testLogInfo)

actionNum.userUpdate.train = getActionTimes(trainMaster$Idx,trainUserupdate)
actionNum.ueseUpdate.test = getActionTimes(testMaster$Idx,testUserupdate)


#----数值型特征离散化---
l = discretizationFeature(train_master_continuity_fillNA,test_master_continuity_fillNA,5)
train_master_continuity_discretization = l[[1]]
test_master_continuity_discretization = l[[2]]
remove(l)

write.csv(train_master_num_discretization,"feature/train_master_num_discretization.csv", row.names = FALSE)
write.csv(test_master_num_discretization,"feature/test_master_num_discretization.csv", row.names = FALSE)



write.csv(train_master_continuity_fillNA, "feature_2w/train_master_continuity_fillNA.csv", row.names = FALSE)
write.csv(train_master_catTocontinuity_fillNA, "feature_2w/train_master_catTocontinuity_fillNA.csv", row.names = FALSE)
write.csv(train_master_strcatLoc_oneHot, "feature_2w/train_master_strcatLoc_oneHot.csv", row.names = FALSE)
write.csv(train_master_strcatNoLoc_oneHot, "feature_2w/train_master_strcatNoLoc_oneHot.csv", row.names = FALSE)


write.csv(test_master_continuity_fillNA, "feature_2w/test_master_continuity_fillNA.csv", row.names = FALSE)
write.csv(test_master_catTocontinuity_fillNA, "feature_2w/test_master_catTocontinuity_fillNA.csv", row.names = FALSE)
write.csv(test_master_strcatLoc_oneHot, "feature_2w/test_master_strcatLoc_oneHot.csv", row.names = FALSE)
write.csv(test_master_strcatNoLoc_oneHot, "feature_2w/test_master_strcatNoLoc_oneHot.csv", row.names = FALSE)