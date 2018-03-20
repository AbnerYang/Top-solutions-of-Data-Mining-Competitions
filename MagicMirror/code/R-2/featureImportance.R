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
            write.csv(data.frame(model_importance_name), paste(paste("xgbImportance/model_importance_", i, sep = ""),".csv",sep = ""),row.names = FALSE)
            print(i)
      }
}


findXgbFeature(iter1 = 1,
               iter2 = 1,
               data = train.x.maxmin,
               target = target,
               param = param,
               round = 1000,
               eta = 1
               )

getMaxMinFeature <- function(data1, data2){
  data = rbind(data1,data2)
  select = c()
  for(i in 1:ncol(data)){
    maxV = max(data[,i])
    minV = min(data[,i])
    if(maxV != minV){
      data1[,i] = (maxV - data1[,i])/(maxV - minV)
      data2[,i] = (maxV - data2[,i])/(maxV - minV)  
      select = append(select,i) 
    }
  }
  l = list(data1[,select],data2[,select])
  return(l)
}

l = getMaxMinFeature(train_467, test_467)
train.x.maxmin = l[[1]]
test.x.maxmin = l[[2]]





train.x.maxmin = round(train.x.2851,4)
test.x.maxmin = round(test.x.2851,4)

write.csv(train.x.maxmin, "xgbfeature/train.x.2851.csv", row.names = FALSE)
write.csv(test.x.maxmin, "xgbfeature/test.x.2851.csv", row.names = FALSE)



train.x.2851 = read.csv("xgbfeature/train.x.2851.csv", stringsAsFactors = FALSE, header = TRUE)
test.x.2851 = read.csv("xgbfeature/test.x.2851.csv", stringsAsFactors = FALSE, header = TRUE)

featureImportance = read.csv("xgbImportance/model_importance_2851.csv",stringsAsFactors = FALSE, header = TRUE)


featureSelect = featureImportance[1:500,1]




model_3528_importance = xgbModelImportant(train_x_select, target, param, 6300, sed)
model_1303_importance = xgbModelImportant(train_LrFeature[,-ncol(train_LrFeature)], target, param, 5500, sed)


model_xgb_3528_importance = getImportanceNames(train_x_select, model_3528_importance)
model_xgb_1303_importance = getImportanceNames(train_LrFeature[,-ncol(train_LrFeature)], model_1303_importance)

write.csv(model_xgb_3528_importance, "model_xgb_3528_importance.csv", row.names = FALSE)
write.csv(model_xgb_1303_importance, "model_xgb_1303_importance.csv", row.names = FALSE)

