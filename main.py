import argparse
import torch
import numpy as np
from Model import Model
from ServerClient import *
from Dataset import Cifar100Dataset

parser = argparse.ArgumentParser(description='Input Operater')
parser.add_argument('--type', type=str, default="server") # or client
parser.add_argument('--ID', type=int, default=1)
parser.add_argument('--host', type=str, default="192.168.1.112")
parser.add_argument('-p', type=str, default="8192")
parser.add_argument("-k", type=int, default=2)
parser.add_argument("--round", type=int, default=5)
args = parser.parse_args()
ID = args.ID
Host_IP = args.host
Host_Port = args.p
Ptype = args.type

batch_size = 128
##########################################
# noniid分布处理 # 陈

# dataset
root="../data/cifar-100-python"
train_set = Cifar100Dataset(root=root, is_train=True, noniid = True) # 引入了noniid参数
val_set = Cifar100Dataset(root=root, is_train=False, noniid = True)
    
train_loader = DataLoader(train_set,
                                shuffle=True,
                                batch_size=batch_size)
val_loader = DataLoader(val_set,shuffle=False,batch_size=batch_size)
    
##########################################

if Ptype == 'client':
    ##########################################
    # 传输参数设置 # 杜

    ##########################################

    # train
    # optimizer = torch.optim.SGD(Model.parameters,lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(Model.parameters,lr=0.03, betas=(0.9, 0.99))
    idnum = ID
    client = Client(Model, optimizer, train_set, idnum, batch_size=batch_size, train_frac=0.8)

if Ptype == 'server':
    ##########################################
    # 传输参数设置 # 杜

    ##########################################

    # train
    # optimizer = torch.optim.SGD(Model.parameters,lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(Model.parameters,lr=0.03, betas=(0.9, 0.99))
    idnum = ID
    server = Server(Model, optimizer, val_set, idnum, batch_size=batch_size, train_frac=0.8)
