# 拍拍贷风控算法竞赛 <br>

#### 竞赛背景 <br>
拍拍贷“魔镜风控系统”从平均400个数据维度评估用户当前的信用状态，给每个借款人打出当前状态的信用分，在此基础上，再结合新发标的信息，打出对于每个标的6个月内逾期率的预测，为投资人提供了关键的决策依据，促进健康高效的互联网金融。拍拍贷首次开放丰富而真实的历史数据，邀你PK“魔镜风控系统”，通过机器学习技术，你能设计出更具预测准确率和计算性能的违约预测算法吗？<br>

#### 竞赛数据 <br>
Master<br>
每一行代表一个样本（一笔成功成交借款），每个样本包含200多个各类字段。<br>
<br>
idx：每一笔贷款的unique key，可以与另外2个文件里的idx相匹配。<br>

UserInfo_*：借款人特征字段<br>

WeblogInfo_*：Info网络行为字段<br>

Education_Info*：学历学籍字段<br>

ThirdParty_Info_PeriodN_*：第三方数据时间段N字段<br>

SocialNetwork_*：社交网络字段<br>

LinstingInfo：借款成交时间<br>

Target：违约标签（1 = 贷款违约，0 = 正常还款）。测试集里不包含target字段。<br>

 
<br>
Log_Info<br>
借款人的登陆信息。<br>

ListingInfo：借款成交时间<br>

LogInfo1：操作代码<br>

LogInfo2：操作类别<br>

LogInfo3：登陆时间<br>

idx：每一笔贷款的unique key<br>

 
<br>
Userupdate_Info<br>
借款人修改信息<br>

ListingInfo1：借款成交时间<br>

UserupdateInfo1：修改内容<br>

UserupdateInfo2：修改时间<br>

idx：每一笔贷款的unique key<br>

#### 评测指标 <br>
AUC<br>
