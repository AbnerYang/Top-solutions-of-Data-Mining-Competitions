#-- Feature Select By Variance---
#==============Parameter===============================
#-- data: process data-[data.frame]
#-- threshold: the value var to select feature-[double]
#==============Return==================================
#-- Integer list : the column of select by var
SelectByVar <- function(data, threshold){
  selectList = c()
  for(i in 1:ncol(data)){
  	print(i)
    if(length(unique(data[,i])) != 1){
      #--tansform data to [0-1]
      data[,i] = (max(data[,i]) - data[,i])/(max(data[,i]) - min(data[,i]))
      #print(head(data[,i]))
      if(var(data[,i]) >= threshold){
        selectList = append(selectList,i)
      }
    }
  }
  return(selectList)
}

#通过特征缺失率选择特征
findRemoveMissFeature <-function(trainNum, missList, base){
  remove = c()
  num = 1
  for(i in 1:ncol(trainNum)){
    if(missList[i] > base){
      remove[num] = i
      num = num + 1
    }
  }
  return(trainNum[,-remove])
}


#---F-Score（非模型评价打分，区别与 F1_score ）是一种衡量特征在两类之间分辨能力的方法.F-score越大说明该特征的辨别能力越强。
findRemovefscoreFeature <-function(data, y, base){
  output = c()
  pos_row = which(y == 1)
  neg_row = which(y == 0)
  for(j in 1:ncol(data)){
    x = data[,j]
    x.pos = data[pos_row,j]
    x.neg = data[neg_row,j]
    #分子
    up = (mean(x.pos) - mean(x))*(mean(x.pos) - mean(x)) +
      (mean(x.neg) - mean(x))*(mean(x.neg) - mean(x))
    pos_sum = 0
    neg_sum = 0
    n_pos = length(x.pos)
    n_neg = length(x.neg)
    for(i in 1:n_pos){
      pos_sum = pos_sum + (x.pos[i] - mean(x.pos)) * (x.pos[i] - mean(x.pos))
    }
    for(i in 1:n_neg){
      neg_sum = neg_sum + (x.neg[i] - mean(x.neg)) * (x.neg[i] - mean(x.neg))
    }
    #分母
    down = 1.0*pos_sum/(n_pos - 1) + 1.0*neg_sum/(n_neg - 1)
    fj = up / down
    print(paste("目前变量：",j))
    output = c(output,fj)
  }
  rank = getRank(output)
  removeIndex = which(rank < base)
  return (removeIndex)
}

#---通过比较特征之间的相关性来筛选出与其他特征相关性高于base(0.9/0.92/0.94/0.96/0.98)的特征做去除处理
findRemoveReleventFeature <-function(trainNum, testNum, base){
  data = rbind(trainNum,testNum)
  # ensure the results are repeatable
  set.seed(7)
  # load the library
  library(mlbench)
  library(caret)
  # load the data
  #data(PimaIndiansDiabetes)
  # calculate correlation matrix
  correlationMatrix <- cor(data)
  # find attributes that are highly corrected (ideally >0.75)
  highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=base)
  # print indexes of highly correlated attributes
  return(highlyCorrelated)
}

