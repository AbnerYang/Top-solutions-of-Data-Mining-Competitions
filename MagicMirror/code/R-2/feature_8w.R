path = "E:/competition_2016/MagicMirror"
setwd(path)
library(xgboost)  
library(Matrix)
library(AUC)
library(ROCR)
library(e1071)
library(LiblineaR)
library(randomForest)
library(hash)


target = read.csv("feature/target.csv", header = TRUE, stringsAsFactors = FALSE)
target = target$target
#----训练集特征

train_master_listingInfo_transform = read.csv("feature_8w/train/train_master_listingInfo_transform.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_UserInfo_weight = read.csv("feature_8w/train/train_master_UserInfo_weight.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_category_weight = read.csv("feature_8w/train/train_master_category_weight.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_all_no_location = read.csv("feature_8w/train/train_master_all_no_location.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_LatAndLong = read.csv("feature_8w/train/train_master_LatAndLong.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_category_num = read.csv("feature_8w/train/train_master_category_num.csv",header = FALSE, stringsAsFactors=FALSE)
train_master_city_rank = read.csv("feature_8w/train/train_master_city_rank.csv",header = FALSE, stringsAsFactors=FALSE)
train_master_missing_scale = read.csv("feature_8w/train/train_master_missing_scale.csv",header = FALSE, stringsAsFactors = FALSE)

train_loginfo_all = read.csv("feature_8w/train/train_loginfo_all.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_time = read.csv("feature_8w/train/train_loginfo_time.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit1 = read.csv("feature_8w/train/train_loginfo_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit3 = read.csv("feature_8w/train/train_loginfo_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit7 = read.csv("feature_8w/train/train_loginfo_limit7.csv",header = FALSE, stringsAsFactors = FALSE)

train_userupdate_all = read.csv("feature_8w/train/train_userupdate_all.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit1 = read.csv("feature_8w/train/train_userupdate_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit3 = read.csv("feature_8w/train/train_userupdate_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit7 = read.csv("feature_8w/train/train_userupdate_limit7.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_time = read.csv("feature_8w/train/train_userupdate_time.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_period_divide = read.csv("feature_8w/train/train_thridParty_period_divide.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_period_subtract = read.csv("feature_8w/train/train_thridParty_period_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_period_sum = read.csv("feature_8w/train/train_thridParty_period_sum.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_section_divide = read.csv("feature_8w/train/train_thridParty_section_divide.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_subtract = read.csv("feature_8w/train/train_thridParty_section_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_sum = read.csv("feature_8w/train/train_thridParty_section_sum.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_period_count = read.csv("feature_8w/train/train_thridParty_period_count.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_count = read.csv("feature_8w/train/train_thridParty_section_count.csv",header = FALSE, stringsAsFactors = FALSE)



#----测试集特征
test_master_listingInfo_transform = read.csv("feature_8w/test/test_master_listingInfo_transform.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_UserInfo_weight = read.csv("feature_8w/test/test_master_UserInfo_weight.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_category_weight = read.csv("feature_8w/test/test_master_category_weight.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_all_no_location = read.csv("feature_8w/test/test_master_all_no_location.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_LatAndLong = read.csv("feature_8w/test/test_master_LatAndLong.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_category_num = read.csv("feature_8w/test/test_master_category_num.csv",header = FALSE, stringsAsFactors=FALSE)
test_master_city_rank = read.csv("feature_8w/test/test_master_city_rank.csv",header = FALSE, stringsAsFactors=FALSE)
test_master_missing_scale = read.csv("feature_8w/test/test_master_missing_scale.csv",header = FALSE, stringsAsFactors = FALSE)

test_loginfo_all = read.csv("feature_8w/test/test_loginfo_all.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_time = read.csv("feature_8w/test/test_loginfo_time.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit1 = read.csv("feature_8w/test/test_loginfo_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit3 = read.csv("feature_8w/test/test_loginfo_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit7 = read.csv("feature_8w/test/test_loginfo_limit7.csv",header = FALSE, stringsAsFactors = FALSE)

test_userupdate_all = read.csv("feature_8w/test/test_userupdate_all.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit1 = read.csv("feature_8w/test/test_userupdate_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit3 = read.csv("feature_8w/test/test_userupdate_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit7 = read.csv("feature_8w/test/test_userupdate_limit7.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_time = read.csv("feature_8w/test/test_userupdate_time.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_period_divide = read.csv("feature_8w/test/test_thridParty_period_divide.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_period_subtract = read.csv("feature_8w/test/test_thridParty_period_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_period_sum = read.csv("feature_8w/test/test_thridParty_period_sum.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_section_divide = read.csv("feature_8w/test/test_thridParty_section_divide.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_subtract = read.csv("feature_8w/test/test_thridParty_section_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_sum = read.csv("feature_8w/test/test_thridParty_section_sum.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_period_count = read.csv("feature_8w/test/test_thridParty_period_count.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_count = read.csv("feature_8w/test/test_thridParty_section_count.csv",header = FALSE, stringsAsFactors = FALSE)




train.x = data.frame(train_thridParty_section_sum[,-1],
					 train_thridParty_section_divide[,-1],
					 train_thridParty_section_subtract[,-1],
					 train_thridParty_period_sum[,-1],
					 train_thridParty_period_divide[,-1],
					 train_thridParty_period_subtract[,-1],
					 train_master_all_no_location[,-1],
					 train_master_category_num[,-1],
					 train_master_category_weight[,-1],
					 train_master_missing_scale[,-1],
					 train_master_listingInfo_transform[,-1],
					 train_master_city_rank[,-1],
					 train_master_UserInfo_weight[,-1],
					 train_master_LatAndLong[,-1],
					 train_loginfo_all[,-1],
					 train_loginfo_time[,-1],
					 train_loginfo_limit1[,-1],
					 train_loginfo_limit3[,-1],
					 train_loginfo_limit7[,-1],
					 train_userupdate_all[,-1],
					 train_userupdate_time[,-1],
					 train_userupdate_limit1[,-1],
					 train_userupdate_limit3[,-1],
					 train_userupdate_limit7[,-1],
					 train_thridParty_period_count[,-1],
					 train_thridParty_section_count[,-1]
					 )


test.x = data.frame(test_thridParty_section_sum[,-1],
					 test_thridParty_section_divide[,-1],
					 test_thridParty_section_subtract[,-1],
					 test_thridParty_period_sum[,-1],
					 test_thridParty_period_divide[,-1],
					 test_thridParty_period_subtract[,-1],
					 test_master_all_no_location[,-1],
					 test_master_category_num[,-1],
					 test_master_category_weight[,-1],
					 test_master_missing_scale[,-1],
					 test_master_listingInfo_transform[,-1],
					 test_master_city_rank[,-1],
					 test_master_UserInfo_weight[,-1],
					 test_master_LatAndLong[,-1],
					 test_loginfo_all[,-1],
					 test_loginfo_time[,-1],
					 test_loginfo_limit1[,-1],
					 test_loginfo_limit3[,-1],
					 test_loginfo_limit7[,-1],
					 test_userupdate_all[,-1],
					 test_userupdate_time[,-1],
					 test_userupdate_limit1[,-1],
					 test_userupdate_limit3[,-1],
					 test_userupdate_limit7[,-1],
					 test_thridParty_period_count[,-1],
					 test_thridParty_section_count[,-1]
					 )

selectList =  noOneValueFeature(train.x)

train_x_select = train.x[,selectList]
test_x_select = test.x[,selectList]

write.csv(train_x_select,"feature_8w/train.x.8w.2485.csv", row.names = FALSE)
write.csv(test_x_select,"feature_8w/test.x.8w.2485.csv", row.names = FALSE)

param = list(booster = 'gbtree',
             max.depth = 3,
             scale_pos_weight=length(which(target == 0))/length(which(target == 1)),
             gamma=0.65,
             lambda=300,
             subsample=0.8,
             colsample_bytree=0.8,
             min_child_weight=4,
             eta=0.02,
             silent = 0, 
             nthread = 8,
             objective='binary:logitraw')


findXgbFeature <- function(iter1, iter2, data, target, param, round, eta){
      for(i in iter1:iter2){
            set.seed(i)
            #indexAll = c(1:ncol(data))
            #indexSelect = sample(indexAll, as.integer(ncol(data)*eta), replace = FALSE)
            model_importance = xgbModelImportant(data, target, param, round, i)
            model_importance_name = getImportanceNames(data, model_importance)
            write.csv(data.frame(model_importance_name), paste(paste("xgbImportance/model_importance_8w_2485_", i, sep = ""),".csv",sep = ""),row.names = FALSE)
            print(i)
      }
}

findXgbFeature(iter1 = 1,
               iter2 = 1,
               data = train_x_select,
               target = target,
               param = param,
               round = 1800,
               eta = 1
               )

train.x = read.csv("feature_8w/train.x.8w.2485.csv",stringsAsFactors = FALSE, header = FALSE, row.names = FALSE)
test.x = read.csv("feature_8w/test.x.8w.2485.csv", stringsAsFactors = FALSE, header = FALSE,row.names = FALSE)

ModelList = c(1)
roundList = c(3000)
cvResult = CVFrameWork(train.x, target, ModelList, roundList, logPath, floor = 1)
