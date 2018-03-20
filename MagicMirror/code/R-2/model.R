library(xgboost)  
library(Matrix)
library(AUC)
library(ROCR)
library(e1071)
library(LiblineaR)
library(randomForest)
library(hash)

#---XGB CV
xgbCVFunction <- function(data, target, name, iter, rou, str,sed){
  train.x.xgb=Matrix(data.matrix(data.frame(data)),sparse=T)
  xgbDLocalTrainAllF=xgb.DMatrix(data=train.x.xgb,label= target)
  auc = c()
  #res = c()
  spw = 12.56
  lam = 700
  subs = 0.7
  colsub = 0.5
  mcw = 5
  maxd = 7
  et = 0.02
  ga = 0.05
  for(i in 1:iter){
    nround = rou
    param = list(booster = 'gbtree',
                 max.depth = maxd,
                 scale_pos_weight=spw,
                 gamma=ga,
                 lambda=lam,
                 subsample=subs,
                 colsample_bytree=colsub,
                 min_child_weight=mcw,
                 eta=et,
                 silent = 0, 
                 nthread = 8,
                 objective='binary:logitraw')
    
    cat('running cross validation\n')
    # do cross validation, this will print result out as
    # [iteration]  metric_name:mean_value+std_value
    # std_value is standard deviation of the metric
    set.seed(sed)
    res = xgb.cv(param, xgbDLocalTrainAllF, nround, metrics={'auc'}, nfold = 4, prediction = TRUE)
    
    
    if(i == 1){
      rr = res$pred
    }else{
      rr = data.frame(rr,res$pred)
    }
    
    pred <- prediction(res$pred, target)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[1] <- c(performance(pred, "auc")@y.values)
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, sed, auc[[1]],rou,str,ga)
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
  }
  return(res$pred)
  #str1 = paste(paste("CVResult/",paste("log",name,sep = ""),sep = ""),".csv",sep = "")
  #write.csv(rr,str1, row.names = FALSE)
}
#---XGB Tree模型 CV
xgbTreeModel <- function(data, target, name, iter, rou, str,sed){
  train.x.xgb=Matrix(data.matrix(data.frame(data)),sparse=T)
  xgbDLocalTrainAllF=xgb.DMatrix(data=train.x.xgb,label= target)
  auc = c()
  #res = c()
  spw = length(which(target == 0))/length(which(target == 1))
  lam = 300
  subs = 0.8
  colsub = 0.8
  mcw = 4
  maxd = 3
  et = 0.02
  ga = 0.65
  for(i in 1:iter){
    nround = rou
    param = list(booster = 'gbtree',
                 max.depth = maxd,
                 scale_pos_weight=spw,
                 gamma=ga,
                 lambda=lam,
                 subsample=subs,
                 colsample_bytree=colsub,
                 min_child_weight=mcw,
                 eta=et,
                 silent = 0, 
                 nthread = 8,
                 objective='binary:logitraw')
    
    cat('running cross validation\n')
    # do cross validation, this will print result out as
    # [iteration]  metric_name:mean_value+std_value
    # std_value is standard deviation of the metric
    set.seed(sed)
    res = xgb.cv(param, xgbDLocalTrainAllF, nround, metrics={'auc'}, nfold = 5, prediction = TRUE)
    
    
    if(i == 1){
      rr = res$pred
    }else{
      rr = data.frame(rr,res$pred)
    }
    
    pred <- prediction(res$pred, target)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[1] <- c(performance(pred, "auc")@y.values)
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, sed, auc[[1]],rou,str,ga)
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
  }
  return(res$pred)
  #str1 = paste(paste("CVResult/",paste("log",name,sep = ""),sep = ""),".csv",sep = "")
  #write.csv(rr,str1, row.names = FALSE)
}
#---XGB Linear模型 CV
xgbLinearModel <- function(data, target, name, iter, rou, str,sed){
  train.x.xgb=Matrix(data.matrix(data.frame(data)),sparse=T)
  xgbDLocalTrainAllF=xgb.DMatrix(data=train.x.xgb,label= target)
  auc = c()
  #res = c()
  spw = length(which(target == 0))/length(which(target == 1))
  lam = 10
  subs = 0.7
  colsub = 0.5
  mcw = 5
  maxd = 7
  et = 0.02
  ga = 0.05
  for(i in 1:iter){
    nround = rou
    param = list(booster = 'gblinear',
                 max.depth = maxd,
                 scale_pos_weight=spw,
                 gamma=ga,
                 lambda=lam,
                 subsample=subs,
                 colsample_bytree=colsub,
                 min_child_weight=mcw,
                 eta=et,
                 silent = 0, 
                 nthread = 8,
                 objective='binary:logitraw')
    
    cat('running cross validation\n')
    # do cross validation, this will print result out as
    # [iteration]  metric_name:mean_value+std_value
    # std_value is standard deviation of the metric
    set.seed(sed)
    res = xgb.cv(param, xgbDLocalTrainAllF, nround, metrics={'auc'}, nfold = 4, prediction = TRUE)
    
    
    if(i == 1){
      rr = res$pred
    }else{
      rr = data.frame(rr,res$pred)
    }
    
    pred <- prediction(res$pred, target)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[1] <- c(performance(pred, "auc")@y.values)
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, sed, auc[[1]],rou,str,ga)
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
  }
  return(res$pred)
  #str1 = paste(paste("CVResult/",paste("log",name,sep = ""),sep = ""),".csv",sep = "")
  #write.csv(rr,str1, row.names = FALSE)
}
#---XGB Tree模型 全集训练预测线上
xgbOnlineTreeModel <- function(data1, data2, target,iter,se, round){
  for(i in 1:iter){
    
    
    train.x.xgb=Matrix(data.matrix(data.frame(data1)),sparse=T)
    test.x.xgb=Matrix(data.matrix(data.frame(data2)),sparse=T)
    
    xgbDTrain=xgb.DMatrix(data=train.x.xgb,label= target)
    xgbDTest= xgb.DMatrix(data=test.x.xgb)
    set.seed(1)
    modelxgb=xgb.train(booster='gbtree',
                       objective='binary:logistic',
                       scale_pos_weight=length(which(target == 0))/length(which(target == 1)),
                       gamma=0.65,
                       lambda=300,
                       subsample=0.8,
                       colsample_bytree=0.8,
                       min_child_weight=4,
                       max_depth=3,
                       eta=0.02,
                       data=xgbDTrain,
                       nrounds=round,
                       metrics='auc',
                       nthread=8,
                       verbose=1,
                       print.every.n = 1
    )
    predXG1=predict(modelxgb,xgbDTrain)
    
    pred <- prediction(predXG1, target)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc <- c(performance(pred, "auc")@y.values)
    print(auc)
    
    predXG=predict(modelxgb,xgbDTest)
    
    if(i == 1){
      predict = data.frame(predXG)
    }else{
      predict = data.frame(predict, predXG)
    }
    
  }
  return(predict)
  #remove(modelxgb)
}
#---XGB Linear模型 全集训练预测线上
xgbOnlineLinearModel <- function(data1, data2, target,iter,se, round){
  for(i in 1:iter){
    
    
    train.x.xgb=Matrix(data.matrix(data.frame(data1)),sparse=T)
    test.x.xgb=Matrix(data.matrix(data.frame(data2)),sparse=T)
    
    xgbDTrain=xgb.DMatrix(data=train.x.xgb,label= target)
    xgbDTest= xgb.DMatrix(data=test.x.xgb)
    set.seed(1)
    modelxgb=xgb.train(booster='gblinear',
                       objective='binary:logistic',
                       scale_pos_weight=12.56,
                       gamma=0.05,
                       lambda=400,
                       subsample=0.7,
                       colsample_bytree=0.5,
                       min_child_weight=5,
                       max_depth=7,
                       eta=0.02,
                       data=xgbDTrain,
                       nrounds=round,
                       metrics='auc',
                       nthread=8
    )
    predXG1=predict(modelxgb,xgbDTrain)
    
    pred <- prediction(predXG1, target)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc <- c(performance(pred, "auc")@y.values)
    print(auc)
    
    predXG=predict(modelxgb,xgbDTest)
    
    if(i == 1){
      predict = data.frame(predXG)
    }else{
      predict = data.frame(predict, predXG)
    }
    
  }
  return(predict)
  #remove(modelxgb)
}
#---LR train test validate 3层测试
LRFloor3Test <- function(data1, data2, data3, target1, target2, target3, iter, seed, name){
  auc = c()
  for(i in 1:iter){
    set.seed(seed)
    model <- LiblineaR(data = data1,verbose = TRUE, target = target1, type = 7)
    pr1 = predict(model,data2, proba =TRUE, decisionValues = TRUE)
    pr2 = predict(model,data3, proba =TRUE, decisionValues = TRUE)
    
    pred <- prediction(pr1$probabilities[,2], target2)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[1] <- c(performance(pred, "auc")@y.values)
    
    pred <- prediction(pr2$probabilities[,2], target3)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[2] <- c(performance(pred, "auc")@y.values)
    
    pred <- prediction(c(pr1$probabilities[,2],pr2$probabilities[,2]), c(target2,target3))
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[3] <- c(performance(pred, "auc")@y.values)
    
    
    
    result = c(auc[[1]],"validate-LR")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    result = c(auc[[2]],"test-LR")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    result = c(auc[[3]],"All-LR")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    
  }
}
#---SVM train test validate 3层测试
SVMFloor3Test <- function(data1, data2, data3, target1, target2, target3, iter, seed, name, rou){
  auc = c()
  for(i in 1:iter){
    set.seed(seed)
    model <- LiblineaR(data = data1,verbose = TRUE, target = target1, type = 1)
    pr1 = predict(model,dat2, proba =TRUE, decisionValues = TRUE)
    pr2 = predict(model,data3, proba =TRUE, decisionValues = TRUE)
    
    pred <- prediction(pr1$decisionValues[,1], target2)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[1] <- c(performance(pred, "auc")@y.values)
    
    pred <- prediction(predTXG, target3)
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[2] <- c(performance(pred, "auc")@y.values)
    
    pred <- prediction(c(predVXG,predTXG), c(target2,target3))
    roc <- performance(pred, "tpr", "fpr")
    plot(roc, main = "ROC chart")
    auc[3] <- c(performance(pred, "auc")@y.values)
    
    
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, seed, auc[[1]],rou,"validate")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, seed, auc[[2]],rou, "test")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    result = c(spw, lam, subs, colsub, mcw, maxd, et, seed, auc[[3]],rou, "All")
    str2 = paste(paste("CVResult/",name,sep = ""),".csv",sep = "")
    write.table(t(result),str2, row.names = FALSE, append = TRUE, sep = ",", col.names = FALSE)
    
    
  }
}
#---XGB 输出训练集特征的重要性排序
xgbModelImportant <- function(data, target,param,round, sed){
  
  train.x.xgb=Matrix(data.matrix(data.frame(data)),sparse=T)
  
  xgbDLocalTrainAllF=xgb.DMatrix(data=train.x.xgb,label= target)
  set.seed(sed)
  
  model = xgb.train(param,xgbDLocalTrainAllF,nrounds = round)
  
  importance = xgb.importance(model = model)
  return(importance)
}


#---lr在线预测---data1为训练集特征，data2为测试集特征， target为目标值， iter循环次数, type指定类型--
#-- type = 0 – L2-regularized logistic regression (primal)
#-- type = 6 - L1-regularized logistic regression
#-- type = 7 – L2-regularized logistic regression (dual)
LROnlinePredict <-function(data1, data2, target, iter, type){
  for(i in 1:10){
    model <- LiblineaR(data = data1,verbose = TRUE, target = target, type = 1)
    
    pr = predict(model,test.x.alllog10, proba =TRUE, decisionValues = TRUE)
    
    predXG = pr$probabilities[,1]
    if(start ==1){
      predFLR = data.frame(predXG)
      start = 2
    }else{
      predFLR = data.frame(predFLR,predXG)
    }
    print(i)
  }
}

#---svm在线预测---data1为训练集特征，data2为测试集特征， target为目标值， iter循环次数, type指定类型--
#-- 1 – L2-regularized L2-loss support vector classification (dual)
#-- 2 – L2-regularized L2-loss support vector classification (primal)
#-- 3 – L2-regularized L1-loss support vector classification (dual)
#-- 4 – support vector classification by Crammer and Singer
#-- 5 – L1-regularized L2-loss support vector classification
SVMOnlinePredict <-function(data1, data2, target, iter, type){
  for(i in 1:10){
    model <- LiblineaR(data = data1,verbose = TRUE, target = target, type = 1)
    
    pr = predict(model,data2, proba =FALSE, decisionValues = TRUE)
    
    predXG = pr$decisionValues[,1]
    if(start ==1){
      predFSVM = data.frame(predXG)
      start = 2
    }else{
      predFSVM = data.frame(predFLR,predXG)
    }
    print(i)
  }
}




