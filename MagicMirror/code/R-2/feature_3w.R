#----训练集特征
train_master_listingInfo_transform = read.csv("feature_3w/train/train_master_listingInfo_transform.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_UserInfo_weight = read.csv("feature_3w/train/train_master_UserInfo_weight.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_category_weight = read.csv("feature_3w/train/train_master_category_weight.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_all_no_location = read.csv("feature_3w/train/train_master_all_no_location.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_LatAndLong = read.csv("feature_3w/train/train_master_LatAndLong.csv",header = FALSE, stringsAsFactors = FALSE)
train_master_category_num = read.csv("feature_3w/train/train_master_category_num.csv",header = FALSE, stringsAsFactors=FALSE)
train_master_city_rank = read.csv("feature_3w/train/train_master_city_rank.csv",header = FALSE, stringsAsFactors=FALSE)
train_master_missing_scale = read.csv("feature_3w/train/train_master_missing_scale.csv",header = FALSE, stringsAsFactors = FALSE)

train_loginfo_all = read.csv("feature_3w/train/train_loginfo_all.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_time = read.csv("feature_3w/train/train_loginfo_time.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit1 = read.csv("feature_3w/train/train_loginfo_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit3 = read.csv("feature_3w/train/train_loginfo_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
train_loginfo_limit7 = read.csv("feature_3w/train/train_loginfo_limit7.csv",header = FALSE, stringsAsFactors = FALSE)

train_userupdate_all = read.csv("feature_3w/train/train_userupdate_all.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit1 = read.csv("feature_3w/train/train_userupdate_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit3 = read.csv("feature_3w/train/train_userupdate_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_limit7 = read.csv("feature_3w/train/train_userupdate_limit7.csv",header = FALSE, stringsAsFactors = FALSE)
train_userupdate_time = read.csv("feature_3w/train/train_userupdate_time.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_period_divide = read.csv("feature_3w/train/train_thridParty_period_divide.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_period_subtract = read.csv("feature_3w/train/train_thridParty_period_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_period_sum = read.csv("feature_3w/train/train_thridParty_period_sum.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_section_divide = read.csv("feature_3w/train/train_thridParty_section_divide.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_subtract = read.csv("feature_3w/train/train_thridParty_section_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_sum = read.csv("feature_3w/train/train_thridParty_section_sum.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_period_divide2 = read.csv("feature_3w/train/train_thridParty_period_divide2.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_period_subtract2 = read.csv("feature_3w/train/train_thridParty_period_subtract2.csv",header = FALSE, stringsAsFactors = FALSE)

train_thridParty_section_divide2 = read.csv("feature_3w/train/train_thridParty_section_divide2.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_subtract2 = read.csv("feature_3w/train/train_thridParty_section_subtract2.csv",header = FALSE, stringsAsFactors = FALSE)


train_thridParty_period_count = read.csv("feature_3w/train/train_thridParty_period_count.csv",header = FALSE, stringsAsFactors = FALSE)
train_thridParty_section_count = read.csv("feature_3w/train/train_thridParty_section_count.csv",header = FALSE, stringsAsFactors = FALSE)



#----测试集特征
test_master_listingInfo_transform = read.csv("feature_3w/test/test_master_listingInfo_transform.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_UserInfo_weight = read.csv("feature_3w/test/test_master_UserInfo_weight.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_category_weight = read.csv("feature_3w/test/test_master_category_weight.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_all_no_location = read.csv("feature_3w/test/test_master_all_no_location.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_LatAndLong = read.csv("feature_3w/test/test_master_LatAndLong.csv",header = FALSE, stringsAsFactors = FALSE)
test_master_category_num = read.csv("feature_3w/test/test_master_category_num.csv",header = FALSE, stringsAsFactors=FALSE)
test_master_city_rank = read.csv("feature_3w/test/test_master_city_rank.csv",header = FALSE, stringsAsFactors=FALSE)
test_master_missing_scale = read.csv("feature_3w/test/test_master_missing_scale.csv",header = FALSE, stringsAsFactors = FALSE)

test_loginfo_all = read.csv("feature_3w/test/test_loginfo_all.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_time = read.csv("feature_3w/test/test_loginfo_time.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit1 = read.csv("feature_3w/test/test_loginfo_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit3 = read.csv("feature_3w/test/test_loginfo_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
test_loginfo_limit7 = read.csv("feature_3w/test/test_loginfo_limit7.csv",header = FALSE, stringsAsFactors = FALSE)

test_userupdate_all = read.csv("feature_3w/test/test_userupdate_all.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit1 = read.csv("feature_3w/test/test_userupdate_limit1.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit3 = read.csv("feature_3w/test/test_userupdate_limit3.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_limit7 = read.csv("feature_3w/test/test_userupdate_limit7.csv",header = FALSE, stringsAsFactors = FALSE)
test_userupdate_time = read.csv("feature_3w/test/test_userupdate_time.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_period_divide = read.csv("feature_3w/test/test_thridParty_period_divide.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_period_subtract = read.csv("feature_3w/test/test_thridParty_period_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_period_sum = read.csv("feature_3w/test/test_thridParty_period_sum.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_section_divide = read.csv("feature_3w/test/test_thridParty_section_divide.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_subtract = read.csv("feature_3w/test/test_thridParty_section_subtract.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_sum = read.csv("feature_3w/test/test_thridParty_section_sum.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_period_divide2 = read.csv("feature_3w/test/test_thridParty_period_divide2.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_period_subtract2 = read.csv("feature_3w/test/test_thridParty_period_subtract2.csv",header = FALSE, stringsAsFactors = FALSE)

test_thridParty_section_divide2 = read.csv("feature_3w/test/test_thridParty_section_divide2.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_subtract2 = read.csv("feature_3w/test/test_thridParty_section_subtract2.csv",header = FALSE, stringsAsFactors = FALSE)


test_thridParty_period_count = read.csv("feature_3w/test/test_thridParty_period_count.csv",header = FALSE, stringsAsFactors = FALSE)
test_thridParty_section_count = read.csv("feature_3w/test/test_thridParty_section_count.csv",header = FALSE, stringsAsFactors = FALSE)




train.x = data.frame(train_thridParty_section_sum[,-1],
					 train_thridParty_section_divide2[,-1],
					 train_thridParty_section_subtract2[,-1],
					 train_thridParty_period_sum[,-1],
					 train_thridParty_period_divide2[,-1],
					 train_thridParty_period_subtract2[,-1],
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
					 test_thridParty_section_divide2[,-1],
					 test_thridParty_section_subtract2[,-1],
					 test_thridParty_period_sum[,-1],
					 test_thridParty_period_divide2[,-1],
					 test_thridParty_period_subtract2[,-1],
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

MissList = findMissPoint(train.x, test.x)
#--获取每列特征除了缺失值外的众数
#ModeList = findModePoint(train.x, test.x, 0)
#--获取每列特征除了缺失值外的中位数
MeanList = findMeanPoint(train.x, test.x, 0)

#--依照Base值填充缺失值
#train_x_fillNA = fillingMiss(train.x, MissList, ModeList, 0.2)
#test_x_fillNA = fillingMiss(test.x, MissList, ModeList, 0.2)

train_x_fillNA = fillingMiss(train.x, MissList, MeanList, 0.2)
test_x_fillNA = fillingMiss(test.x, MissList, MeanList, 0.2)

write.csv(train.x,"feature_3w/train.x.3w.2693.csv", row.names = FALSE)
write.csv(test.x,"feature_3w/test.x.3w.2693.csv", row.names = FALSE)

selectList =  noOneValueFeature(train.x)

train_x_select = train.x[,selectList]
test_x_select = test.x[,selectList]
write.csv(train_x_select,"feature_3w/train.x.3w.4747.csv", row.names = FALSE)
write.csv(test_x_select,"feature_3w/test.x.3w.4747.csv", row.names = FALSE)





