Personalized Federated Learning
=

## 工作流程

1. 服务器->端侧 模型+聚类id+端独自训练epoch数
2. 端侧 训练
3. 端侧->服务器 发送模型和梯度
4. 服务器 把接收到的模型+梯度存到每个分组对应的数据结构里
5. 服务器 当一个分组的client全部训练完的时候，发送给聚类和平均的模块。

## 模块

1. ServerClient.py: 服务端和端的类
2. Clustering.py: 聚类
3. Model.py: 模型
4. Train_device.py: 模型精度测量
5. Dataset.py: 数据接口（加入了noniid的参数，但是没有实现）
6. main.py: 主

## Requirements

python-socketio
websocket-client
eventlet
