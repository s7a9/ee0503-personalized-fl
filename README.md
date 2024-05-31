Personalized Federated Learning
=

## 工作流程

1. 服务器->端侧 模型+聚类id+端独自训练epoch数
2. 端侧 训练
3. 端侧->服务器 发送模型和梯度
4. 服务器 把接收到的模型+梯度存到每个分组对应的数据结构里
5. 服务器 当一个分组的client全部训练完的时候，发送给聚类和平均的模块。


## 聚类和平均模块
Guoqinlu：安排了基本的模块暂时

## 传输
Guoqinlu：main里面加入了相应的IP等等的预制，可以参考着改一下；

## noniid
Guoqinlu: main里面的dataload，可以实现在dataset.py里，具体我在助教的基础上加了一个noniid=True参数；

