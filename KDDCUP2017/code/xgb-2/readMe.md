# ReadMe

#### feature.py中的路径有：<br>
route_in_path = 'data/routes (table 4).csv' #输入文件路径<br>
train_in_path = 'data/train.csv' #输入文件路径<br>
test_in_path = 'data/test.csv' #输入文件路径<br>

train_out_x_path = 'data/train.x.tsv' #输出文件路径<br>
train_out_y_path = 'data/train.y.tsv' #输出文件路径<br>

test_out_x_path = 'data/test.x.tsv' #输出文件路径<br>

其中需要注意的是train.csv 和 test.csv的表头一定是：intersection_id,tollgate_id,vehicle_id,starting_time,travel_seq,travel_time<br>





#### load.py的文件路径有：<br>
link_path = 'data/links (table 3).csv' #输入文件路径<br>




配置好文件的输入输出路径后，先运行feature.py然后再运行xgb.py就可以得到最终结果了<br>
