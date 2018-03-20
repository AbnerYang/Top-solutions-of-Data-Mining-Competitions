#--Fill the NA value with num value
#=================Parameters======================
#-- data: process data -[data.frame]
#-- NAType1: the string type of NA -[list character]
#-- Value: the fill value -[double,integer]
#=================Return==========================
#-- data: processed data -[data.frame]
FillNAValue <- function(data, NAType1, value){
	for(i in 1:ncol(data)){
		if(is.character(data[,i])){
			for(j in 1:length(NAType1)){
				data[which(data[,i] == NAType1[j]),i] = value
			}
		}else{
			data[which(is.na(data[,i])),i] = value
		}
	}
	return(data)
}

#---第三方相邻period的差特征
getTrendFeature <- function(data){
  for(i in 2:7){
    p1 = i-1
    p2 = i
    for(j in 1:17){
      index1 = (p1-1)*17+j
      index2 = (p2-1)*17+j
      if(i == 2 & j == 1){
        feature = c(data[,index2] - data[,index1])
      }else{
        f = c(data[,index2] - data[,index1])
                feature = data.frame(feature,f)
      }
    }
  }
  return(feature)
}


#--Get The one-hot encoder
#=================Parameters======================
#-- train: train process data -[data.frame]
#-- test: test process data -[data.frame]
#=================Return==========================
#-- l: list contains two data.frame -[list]
getOneHotFeature <- function(train,test){
  for(i in 1:ncol(train)){
    l = unique(c(train[,i],test[,i]))
    matrix1 = matrix(0,nrow = nrow(train),ncol = length(l),byrow = TRUE)
    matrix2 = matrix(0,nrow = nrow(test),ncol = length(l),byrow = TRUE)
    
    for(j in 1:length(l)){
      matrix1[which(train[,i] == l[j]),j] = 1
      matrix2[which(test[,i] == l[j]),j] = 1
    }
    
    if(i == 1){
      oneHot1 = data.frame(matrix1)
      oneHot2 = data.frame(matrix2)
    }else{
      oneHot1 = data.frame(oneHot1, matrix1)
      oneHot2 = data.frame(oneHot2, matrix2)
    }
    print(i)
  }
  l = list(oneHot1,oneHot2)
  return(l)
}


#--Get The density Matrix encoder
#=================Parameters======================
#-- train: train process data -[data.frame]
#-- test: test process data -[data.frame]
#=================Return==========================
#-- l: list contains two data.frame -[list]
getDensityFeature <- function(train,test){
  for(i in 1:ncol(train)){
    l = unique(c(train[,i],test[,i]))  
    for(j in 1:length(l)){
      p = (length(which(train[,i] == l[j]))+length(which(test[,i] == l[j])))/(nrow(train)+nrow(test))
      train[which(train[,i] == l[j]),i] = p
      test[which(test[,i] == l[j]),i] = p
    }
    train[,i] = as.double(train[,i])
    test[,i] = as.double(test[,i])
    print(i)
  }
  l = list(train,test)
  return(l)
}

#--Get The Num Stat feature
#=================Parameters======================
#-- data: process data -[data.frame]
#=================Return==========================
#-- matrix: stat data frame -[data.frame]
getNumStatFeature <- function(data){
	matrix1 = matrix(0, nrow = nrow(data), ncol = 12, byrow = TRUE)
	for(i in 1:nrow(data)){
		matrix1[i,1]  = length(which(data[i,] == -1))
		matrix1[i,2]  = mean(as.numeric(data[i,]))
		a = as.numeric(data[i,])
		b = as.integer(quantile(a, c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)))
		matrix1[i,c(3:12)] = b
		print(i)
	}
	return(data.frame(matrix1))
}

getLocExtraFeature <- function(data){
	matrix1 = matrix(0, nrow = nrow(data), ncol = 8, byrow = TRUE)
	for(i in 1:nrow(data)){
		if(data[i,1] == data[i,2]){
			matrix1[i,1] = 1
		}else{
			matrix1[i,1] = 0
		}
		if(data[i,1] == data[i,4]){
			matrix1[i,2] = 1
		}else{
			matrix1[i,2] = 0
		}
		if(data[i,1] == data[i,6]){
			matrix1[i,3] = 1
		}else{
			matrix1[i,3] = 0
		}
		if(data[i,2] == data[i,4]){
			matrix1[i,4] = 1
		}else{
			matrix1[i,4] = 0
		}
		if(data[i,2] == data[i,6]){
			matrix1[i,5] = 1
		}else{
			matrix1[i,5] = 0
		}
		if(data[i,4] == data[i,6]){
			matrix1[i,6] = 1
		}else{
			matrix1[i,6] = 0
		}
		if(data[i,3] == data[i,5]){
			matrix1[i,7] = 1
		}else{
			matrix1[i,7] = 0
		}
		if(data[i,7] == "D"){
			matrix1[i,8] = 0
		}else{
			matrix1[i,8] = 1
		}
		print(i)
	}
	return(data.frame(matrix1))
}

#--获取每个特征的缺失率
findMissPoint <- function(dataTrain, dataTest){
  miss = c()
  data = rbind(dataTrain, dataTest)
  for(i in 1:ncol(data)){
    missNum = length(data[which(data[,i] == -1),i])
    miss[i] = missNum/nrow(data)
  }
  return(miss)
}
#---填充缺失值
fillingMiss <-function(dataTrain, misslist, fitlist, base){
  #--转换-1
  changeValue = -1
  for(i in 1:ncol(dataTrain)){
    if(misslist[i] <= base){
      dataTrain[which(dataTrain[,i] == changeValue),i] = fitlist[i]
    }
  }
  return(dataTrain)
}

#---寻找每个特征的众数
findModePoint <- function(dataTrain, dataTest, type){
  data = rbind(dataTrain,dataTest)
  mode = c()
  #---除去离群点后找众数
  if(type == 1){
    for(i in 1:ncol(data)){
      maxTest = max(dataTest[,i])
      list = data[which(data[,i] <= maxTest),i]
      tableList = table(list)
      mode[i] = as.numeric(names(which.max(tableList)))
    }
  }else{ #--除去-1值后找众数
    for(i in 1:ncol(data)){
      list = data[which(data[,i] != "-1"),i]
      tableList = table(list)
      mode[i] = as.numeric(names(which.max(tableList)))
      print(i)
    }
  }
  return(mode)
}

#---清理数据形式上的异常
cleanCatTypeFeature <- function(data){
  for(i in 1:ncol(data)){
    if(is.character(data[,i])){
      data[,i] = gsub(" ","",data[,i])
      data[,i] = gsub("省","",data[,i])
      data[,i] = gsub("自治州","",data[,i])
      data[,i] = gsub("自治区","",data[,i])
      data[,i] = gsub("市","",data[,i])
      data[,i] = gsub("区","",data[,i])
      data[,i] = gsub("县","",data[,i])
      data[,i] = gsub("旗","",data[,i])
      data[,i] = gsub("盟","",data[,i])
      data[which(is.na(data[,i])),i] = "不详"
    }
  }
  return(data)
}

#--证明数值型特征NA和-1不同时存在
findNegOneWithNAValue <- function(data){
	k1 = c()
	k2 = c()
	for(i in 1:ncol(data)){
		if(length(which(data[,i] == -1)) > 0){
			k1 = append(k1,i)
		}
		if(length(which(is.na(data[,i]))) > 0){
			k2 = append(k2,i)
		}
	}
	print(intersect(k1,k2))
}

#--找到类别型特征中为字符串类别的列
findStrList <- function(data){
	k = c()
	for(i in 1:ncol(data)){
		if(is.character(data[,i])){
			k = append(k,i)
		}
	}
	return(k)
}
#--获取Miss信息的统计特征
getMissStat <- function(train_master_num, test_master_num){
	list1 = c()
	list2 = c()
	for(i in 1:nrow(train_master_num)){
		list1[i] = length(which(train_master_num[i,] == -1))
		print(i)
	}
	for(i in 1:nrow(test_master_num)){
		list2[i] = length(which(test_master_num[i,] == -1))
		print(i)
	}
	missStat = c(list1,list2)
	q = c(0,as.integer(quantile(missStat,c(0.25,0.5,0.75,1))))

	matrix1 = matrix(0, nrow = nrow(train_master_num), ncol = 5, byrow = TRUE)
	matrix2 = matrix(0, nrow = nrow(test_master_num), ncol = 5, byrow = TRUE)

	matrix1[,1] = list1
	matrix2[,1] = list2

	for(i in 1:(length(q)-1)){
		matrix1[which((list1 >= q[i]) & (list1 < q[i+1])),i+1] = 1
		matrix2[which((list2 >= q[i]) & (list2 < q[i+1])),i+1] = 1
	}
	ll = list(data.frame(matrix1), data.frame(matrix2))
	return(ll)
}

#--找出数据每一维特征的unique变量个数--
findCatInNumByUniqueInfo <- function(data1,data2,base){
  data = rbind(data1,data2)
  catList = c()
  for(i in 1:ncol(data)){
    k = length(unique(data[,i]))
    if(k <= base){
      catList = append(catList,i)
    }
  }
  onehot = getOneHotFeature(data1[,catList], data2[,catList])
  return(onehot)
}
#--获取每个特征的中位数
findMedianPoint <- function(dataTrain, dataTest, type){
  data = rbind(dataTrain,dataTest)
  median = c()
  #---除去离群点后找中位数
  if(type == 1){
    for(i in 1:ncol(data)){
      maxTest = max(dataTest[,i])
      list = data[which(data[,i] <= maxTest),i]
      median[i] = median(list)
    }
  }else{ #--除去-1值后找中位数
    for(i in 1:ncol(data)){
      list = data[which(as.integer(data[,i]) != -1),i]
      median[i] = median(list)
    }
  }
  return(median)
}

#--3w训练2w预测集在daily和final以及总的平均AUC得分
findOnlineTestResult <- function(predict, testIdx, daily, final){
  dailyIndex = c()
  for(i in 1:nrow(daily)){
    dailyIndex = append(dailyIndex, which(daily[i,1] == testIdx))
  }

  finalIndex = c()
  for(i in 1:nrow(final)){
    finalIndex = append(finalIndex, which(final[i,1] == testIdx))
  }

  cat("Daily Score:")
  getAUC(predict[dailyIndex], daily[,2])
  cat("Final Score:")
  getAUC(predict[finalIndex], final[,2])
  cat("All Score:")
  getAUC(c(predict[dailyIndex],predict[finalIndex]),c(daily[,2], final[,2]))

}
#--获取每个特征的均值
findMeanPoint <- function(dataTrain, dataTest, type){
  data = rbind(dataTrain,dataTest)
  mean = c()
  #---除去离群点后找中位数
  if(type == 1){
    for(i in 1:ncol(data)){
      maxTest = max(dataTest[,i])
      list = data[which(data[,i] <= maxTest),i]
      mean[i] = mean(list)
    }
  }else{ #--除去-1值后找中位数
    for(i in 1:ncol(data)){
      list = data[which(as.integer(data[,i]) != -1),i]
      mean[i] = mean(list)
    }
  }
  return(mean)
}

#--获取特征的排名
getRankFeature <- function(data1,data2){
	data = rbind(data1,data2)
	for(i in 1:ncol(data)){
		a = log(rank(data[,i]),10)
		if(i == 1){
			re = data.frame(a)
		}else{
			re = data.frame(re,a)
		}
	}
	train = re[1:nrow(data1),]
	test = re[(nrow(data1)+1):(nrow(data1)+nrow(data2)),]
	l = list(train,test)
	return(l)
}

#--划分训练集
splitTrainTestData <- function(trainData, train_target, split, seed){
  index = c(1:nrow(trainData))
  indexPositive = which(train_target == 1)
  indexNegative = which(train_target == 0)
  set.seed(seed)
  localTrainIndex0 = sample(indexNegative, as.integer(split*length(indexNegative)), replace = FALSE)
  localTrainIndex1 = sample(indexPositive, as.integer(split*length(indexPositive)), replace = FALSE)
  
  localTrainIndex = c(localTrainIndex0,localTrainIndex1)
  localTrainIndex = sample(localTrainIndex, length(localTrainIndex), replace = FALSE)
  
  localTestIndex = setdiff(index, localTrainIndex)

  l = list(trainData[localTrainIndex,], trainData[localTestIndex,], train_target[localTrainIndex], train_target[localTestIndex])
  return(l)
}

#--获取AUC
getAUC <- function(pre, target){
  pred <- prediction(pre, target)
  roc <- performance(pred, "tpr", "fpr")
  plot(roc, main = "ROC chart")
  auc <- c(performance(pred, "auc")@y.values)
  print(auc)
  return(auc[[1]])
}

#--获取不同结果Blend的AUC
getBlendAuc <- function(data, target, weight){
  result = rep(0, nrow(data))
  for(i in 1:ncol(data)){
    data[,i] = (data[,i]-min(data[,i]))/(max(data[,i]) - min(data[,i]))
    result = result + data[,i]*weight[i]
  }
  pred <- prediction(result, target)
  roc <- performance(pred, "tpr", "fpr")
  plot(roc, main = "ROC chart")
  auc <- c(performance(pred, "auc")@y.values)
  print(auc)
  return(result)
}

#--XGB CV框架
CVFrameWork <-function(localTrain, target, ModelTypeList, roundList, logPath, floor){

  for(i in 1:length(ModelTypeList)){
    if(ModelTypeList[i] == 1){
      logInfo(logPath,c("本地训练集第",floor,"层第 ",i," 个模型cv: gbtree-postive train"))
      if(i == 1){
        pre = xgbTreeModel(localTrain, target, "cvfind",1, roundList[i],"gbtree-postive", 1)
        cvResult = pre
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = xgbTreeModel(localTrain, target, "cvfind",1, roundList[i],"gbtree-postive", 1)
        cvResult = data.frame(cvResult,pre)
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 2){
      logInfo(logPath,c("本地训练集第",floor,"层第 ",i," 个模型cv: gbtree-negative train"))
      if(i == 1){
        pre = 1-xgbTreeModel(localTrain, 1-target, "cvfind",1, roundList[i],"gbtree-negative", 1)
        cvResult = pre
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = 1-xgbTreeModel(localTrain, 1-target, "cvfind",1, roundList[i],"gbtree-negative", 1)
        cvResult = data.frame(cvResult,pre)
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 3){
      logInfo(logPath,c("本地训练集第",floor,"层第 ",i," 个模型cv: gblinear-positive train"))
      if(i == 1){
        pre = xgbLinearModel(localTrain, target, "cvfind",1, roundList[i],"gbtree-positive", 1)
        cvResult = pre
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = xgbLinearModel(localTrain, target, "cvfind",1, roundList[i],"gbtree-positive", 1)
        cvResult = data.frame(cvResult,pre)
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 4){
      logInfo(logPath,c("本地训练集第",floor,"层第 ",i," 个模型cv: gblinear-negative train"))
      if(i == 1){
        pre = 1-xgbLinearModel(localTrain, 1-target, "cvfind",1, roundList[i],"gblinear-negative", 1)
        cvResult = pre
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = 1-xgbLinearModel(localTrain, 1-target, "cvfind",1, roundList[i],"gblinear-negative", 1)
        cvResult = data.frame(cvResult,pre)
        logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }
  }
  return(cvResult)
}

#--XGB Online提交框架
OnlineFramework <- function(data1, data2, target, ModelTypeList, roundList, logPath, floor){

  for(i in 1:length(ModelTypeList)){
    if(ModelTypeList[i] == 1){
      logInfo(logPath,c("在线训练集第",floor,"层第 ",i," 个模型cv: gbtree-postive train"))
      if(i == 1){
        pre = xgbOnlineTreeModel(localTrain, localTest, target, 1, 1, roundList[i])
        onlineResult = pre
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = xgbOnlineTreeModel(localTrain, localTest, target, 1, 1, roundList[i])
        onlineResult = data.frame(onlineResult,pre)
       # logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 2){
      logInfo(logPath,c("在线训练集第",floor,"层第 ",i," 个模型cv: gbtree-negative train"))
      if(i == 1){
        pre = 1-xgbOnlineTreeModel(localTrain, localTest, 1-target, 1, 1, roundList[i])
        onlineResult = pre
       # logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = 1-xgbOnlineTreeModel(localTrain, localTest, 1-target, 1, 1, roundList[i])
        onlineResult = data.frame(onlineResult,pre)
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 3){
      logInfo(logPath,c("在线训练集第",floor,"层第 ",i," 个模型cv: gblinear-positive train"))
      if(i == 1){
        pre = xgbOnlineLinearModel(localTrain, localTest, target, 1, 1, roundList[i])
        onlineResult = pre
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = xgbOnlineLinearModel(localTrain, localTest, target, 1, 1, roundList[i])
        onlineResult = data.frame(onlineResult,pre)
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    if(ModelTypeList[i] == 4){
      logInfo(logPath,c("在线训练集第",floor,"层第 ",i," 个模型cv: gblinear-negative train"))
      if(i == 1){
        pre = 1-xgbOnlineLinearModel(localTrain, localTest, 1-target, 1, 1, roundList[i])
        onlineResult = pre
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }else{
        pre = 1-xgbOnlineLinearModel(localTrain, localTest, 1-target, 1, 1, roundList[i])
        onlineResult = data.frame(onlineResult,pre)
        #logInfo(logPath,c("AUC: ", getAUC(pre, target)))
      }
    }

    
  }
  return(onlineResult)
}

#----log信息
logInfo <- function(logPath,listString){
  op <- options(digits.secs = 6)
  str = NULL
  for(i in 1:length(listString)){
    str = paste(str,listString[i],sep = "")
  }
  #--ouput log info to file--
  cat(paste(Sys.time(), str, sep = " ---- "), file = logPath, append = TRUE, sep = "\n")
  #--output log info to console--
  print(paste(Sys.time(), str, sep = " ---- "))
}  

#---设置Log信息
setLogInfo <- function(workPath){
  #--set the work path--
  setwd(workPath)
  #--process time info to log--
  startTime = unlist(strsplit(as.character(Sys.time())," "))
  date = unlist(strsplit(as.character(startTime[[1]]),"-"))
  time = unlist(strsplit(as.character(startTime[[2]]),":"))
  d = ""
  t = ""
  for(i in 1:3){
    d = paste(d,date[[i]],sep = "")
    t = paste(t,time[[i]],sep = "")
  }
  strtime = paste(d,t,sep = "_")
  print(strtime)
  logStr = paste("log/logInfo_",strtime,sep = "")
  logPath = paste(logStr,".txt",sep = "")
  return(logPath)
}

#--获取LogInfo 各种操作类型的次数比重
getLogInfoNumAndBitFeature <- function(uid,data,time){
  index = c()
  for(i in 1:nrow(data)){
    day = getDays(as.integer(gsub("-","",data[i,2])), as.integer(gsub("-","",data[i,5])))
    if(day <= time){
      index = append(index,i)
    }
    print(i)
  }
  dataSelect = data[index,]
  list1 = unique(dataSelect[,3])
  list2 = unique(dataSelect[,4])
  
  type1 = as.character(list1)
  typeIndex1 = c(1:length(type1))
  names(typeIndex1) = type1 
  
  type2 = as.character(list2)
  typeIndex2 = c(1:length(type2))
  names(typeIndex2) = type2
  
  matrix1 = matrix(0, nrow = length(uid), ncol = 2*(length(list1)+length(list2))+1, byrow = TRUE)
  
  for(i in 1:length(uid)){
    result1 = findStringList(dataSelect[which(dataSelect[,1] == uid[i]),3], typeIndex1)
    matrix1[i,c(2:(1+length(list1)))] = result1
    matrix1[i,c((2+length(list1)):(1+2*length(list1)))] = result1/length(which(dataSelect[,1] == uid[i]))
    
    result2 = findStringList(dataSelect[which(dataSelect[,1] == uid[i]),4], typeIndex2)
    matrix1[i,c((2+2*length(list1)):(1+2*length(list1)+length(list2)))] = result2
    matrix1[i,c((2+2*length(list1)+length(list2)):(1+2*length(list1)+2*length(list2)))] = result2/length(which(dataSelect[,1] == uid[i]))
    
    matrix1[i,1] = length(which(dataSelect[,1] == uid[i]))
    print(i)
  }
  
  return(data.frame(matrix1))
  
}
#--获取UserUpdate 各种操作类型的次数比重
getUserUpdateFeature <- function(trainUid, testUid, data1, data2){
  type = unique(c(data1[,3],data2[,3]))
  typeIndex = c(1:length(type))
  names(typeIndex) = type 
  
  for(i in 1:length(trainUid)){
    result = findStringList(data1[which(data1[,1] == trainUid[i]),3], typeIndex)
    if(length(unique(result)) == 1){
      result = rep(-1, length(result))
    }
    if(i == 1){
      matrixF1 = result 
    }else{
      matrixF1 = rbind(matrixF1,result)
    }
    print(i)
  }
  print("Train over!")
  for(j in 1:length(testUid)){
    result = findStringList(data2[which(data2[,1] == testUid[j]),3], typeIndex)
    if(length(unique(result)) == 1){
      result = rep(-1, length(result))
    }
    if(j == 1){
      matrixF2 = result
    }else{
      matrixF2 = rbind(matrixF2,result)
    }
    print(j)
  }
  print("Test over!")
  matrixF1 = data.frame(matrixF1)
  matrixF2 = data.frame(matrixF2)
  l = list(matrixF1,matrixF2)
  return(l)
}

#--获取LogInfo 各种操作类型的是否存在
getLoginfoFeature <- function(trainUid, testUid, data1, data2, kk){
  type = as.character(unique(c(data1[,kk],data2[,kk])))
  typeIndex = c(1:length(type))
  names(typeIndex) = type 
  
  
  for(i in 1:length(trainUid)){
    result = findStringList(data1[which(data1[,1] == trainUid[i]),kk], typeIndex)
    if(length(unique(result)) == 1){
      result = rep(-1, length(result))
    }
    if(i == 1){
      matrixF1 = result
    }else{
      matrixF1 = rbind(matrixF1,result)
    }
    #print(i)
  }
  print("Train over!")
  for(j in 1:length(testUid)){
    result = findStringList(data2[which(data2[,1] == testUid[j]),kk], typeIndex)
    if(length(unique(result)) == 1){
      result = rep(-1, length(result))
    }
    if(j == 1){
      matrixF2 = result
    }else{
      matrixF2 = rbind(matrixF2,result)
    } 
    #print(j)
  }
  print("Test over!")
  matrixF1 = data.frame(matrixF1)
  matrixF2 = data.frame(matrixF2)
  l = list(matrixF1,matrixF2)
  return(l)
}
#--获取用户操作条数
getActionTimes <- function(uid, data){
  n = c()
  for(i in 1:length(uid)){
    n[i] = length(data[which(data[,1] == uid[i]),1])
    if(n[i] == 0){
      n[i] == -1
    }
  }
  return(n)
}
#--获取用户操作向量
findStringList <- function(data, index){
  num = rep(0,length(index))
  for(i in 1:length(data)){
    l = as.integer(index[data[i]])
    num[l] = num[l] + 1
  }
  return(num)
}
#--获取用户userupdate操作时间特征
getUserUpdateTimeFeature <- function(trainUid,data){
  t = c()
  for(i in 1:length(trainUid)){
    dateInfo1 = data[which(data[,1] == trainUid[i]),2]
    dateInfo2 = data[which(data[,1] == trainUid[i]),4]
    
    if(length(dateInfo1) == 0){
      t[i] = -1
    }else{
      time1 = as.integer(gsub("/","",dateInfo1[1]))
      time2 = max(as.integer(gsub("/","",dateInfo2)))
      
      t[i] = getDays(time1,time2)
    }
    
    print(i)
  }
  return(t)
}
#--获取类别型特征的TF_IDF特征
getTFIDFFeature <- function(uid1, uid2, data1, data2, k){
  
  hashMap = findHashMap(data1,data2,k)
  Index = names(hashMap)
  D = length(unique(data1[,1]))+length(unique(data2[,1]))
  
  matrix1 = matrix(0, nrow = length(uid1), ncol = length(hashMap), byrow = TRUE)
  matrix2 = matrix(0, nrow = length(uid2), ncol = length(hashMap), byrow = TRUE)
  
  for(i in 1:length(uid1)){
    l = data1[which(data1[,1] == uid1[i]),k]
    if(length(l) != 0){
      wordSet1 = table(l)
      name = names(wordSet1)
      for(j in 1:length(wordSet1)){
        tf = as.integer(wordSet1[name[j]])/length(l)
        idf = log(D/as.integer(hashMap[name[j]]), base = 10)
        matrix1[i,which(Index == name[j])] = tf*idf
      }
      print(i)
    }
  }
  
  for(i in 1:length(uid2)){
    l = data2[which(data2[,1] == uid2[i]),k]
    if(length(l) != 0){
      wordSet2 = table(l)
      name = names(wordSet2)
      for(j in 1:length(wordSet2)){
        tf = as.integer(wordSet2[name[j]])/length(l)
        idf = log(D/as.integer(hashMap[name[j]]), base = 10)
        matrix2[i,which(Index == name[j])] = tf*idf
      }
      print(i)
    }
  }
  
  matrix1 = data.frame(matrix1)
  matrix2 = data.frame(matrix2)
  
  ll = list(matrix1,matrix2)
  return(ll)
  
}
#--HashMap  R
findHashMap <- function(data1, data2, k){
  l = unique(c(data1[,k], data2[,k]))
  num = rep(0,length(l))
  for(i in 1:length(l)){
    num[i] = length(unique(data1[which(data1[,k] == l[i]),1]))
    num[i] = num[i] + length(unique(data2[which(data2[,k] == l[i]),1]))
  }
  names(num) = l
  return(num)
}

#--多特征做log变换
getlog <- function(data){
  for(i in 1:ncol(data)){
    data[which(data[,i] > 0),i] = log(data[which(data[,i]>0),i],10)
  }
  return(data)
}

#--离散化数值变量特征
discretizationFeature <- function(data1, data2, num){
  data = rbind(data1,data2)
  k = c(1:num)/num
  for(i in 1:ncol(data)){
    listWeight = unique(as.integer(quantile(data[,i],k)))
    listWeight = c(min(data[,i])-1,listWeight)
    matrix1 = matrix(0,nrow = nrow(data), ncol = length(listWeight)-1, byrow = TRUE)
    for(j in 1:(length(listWeight)-1)){
      matrix1[which((data[,i] > listWeight[j]) & (data[,i] <= listWeight[j+1])),j] = 1
    }
    if(i == 1){
      result = data.frame(matrix1)
    }else{
      result = data.frame(result,data.frame(matrix1))
    }
    print(i)
  }
  l = list(data.frame(result[1:nrow(data1),]), data.frame(result[(nrow(data1)+1):nrow(data),]))
  return(l)
}
#--Rank AUC融合
getRankBlendAuc <- function(data, target, weight){
  result = rep(0, nrow(data))
  for(i in 1:ncol(data)){
    data[,i] = rank(data[,i])
    result = result + data[,i]*weight[i]
  }
  pred <- prediction(result, target)
  roc <- performance(pred, "tpr", "fpr")
  plot(roc, main = "ROC chart")
  auc <- c(performance(pred, "auc")@y.values)
  print(auc)
  return(result)
}
#--获取特征index对应的名字
getImportanceNames <- function(data, imp){
  name = names(data)
  imp = data.frame(imp)
  for(i in 1:nrow(imp)){
      imp[i,1] = name[as.integer(imp[i,1])+1]
  }
  return(imp)
} 

#--找到特征中不只含一个值的特征
noOneValueFeature <- function(data){
  listR = c()
  for(i in 1:ncol(data)){
    if(length(unique(data[,i])) != 1){
      listR = append(listR,i)
    }
  }
  return(listR)
}
#--归一化特征
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
#--theta倍的标准差判断数据异常
processOutiler <- function(data, theta){
  for(i in 1:ncol(data)){
    mean = mean(data[which(!is.na(data[,i])),i])
    sd = sd(data[which(!is.na(data[,i])),i])
    data[which(data[,i] >= (sd*theta+mean)),i] = sd*theta+mean
    print(i)
  }
  return(data)
}