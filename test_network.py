import argparse
import torch
import numpy as np
from Model import Model
from ServerClient import *
from Dataset import Cifar100Dataset
from network.server import NetworkServer
from network.client import run_client, send_data
from utils.serialize import model_to_bytes, bytes_to_model

parser = argparse.ArgumentParser(description='Input Operater')
parser.add_argument('--type', type=str, default="server") # or client
parser.add_argument('--ID', type=int, default=1)
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('-p', type=int, default="8192")
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
# root="../data/cifar-100-python"
# train_set = Cifar100Dataset(root=root, is_train=True, noniid = True) # 引入了noniid参数
# val_set = Cifar100Dataset(root=root, is_train=False, noniid = True)
    
# train_loader = DataLoader(train_set,
#                                 shuffle=True,
#                                 batch_size=batch_size)
# val_loader = DataLoader(val_set,shuffle=False,batch_size=batch_size)
    
##########################################

if Ptype == 'client':
    # train
    # optimizer = torch.optim.SGD(Model.parameters,lr=0.1, momentum=0.9)
    # optimizer = torch.optim.Adam(Model.parameters,lr=0.03, betas=(0.9, 0.99))
    # idnum = ID
    # client = Client(Model, optimizer, train_set, idnum, batch_size=batch_size, train_frac=0.8)

    ##########################################
    # 传输参数设置 # 杜
    def train_callback(sio, data): # receive data from server
        # TODO: Train the model, then send updated model to server
        print('=====================')
        print('Training model')
        trained_data = data + ' trained'
        sio.sleep(random.randint(1, 4)) # remove after testing
        print('Sending data')
        send_data(sio, trained_data)

    run_client(Host_Port, train_callback, Host_IP)
    ##########################################

elif Ptype == 'server':
    # train
    # optimizer = torch.optim.SGD(Model.parameters,lr=0.1, momentum=0.9)
    # optimizer = torch.optim.Adam(Model.parameters,lr=0.03, betas=(0.9, 0.99))
    # idnum = ID
    # server = Server(Model, optimizer, val_set, idnum, batch_size=batch_size, train_frac=0.8)


    ##########################################
    # 传输参数设置 # 杜
    def group_complete_callback(group):
        """
        When all clients in a group have finished training, this function is called.
        """
        print('=====================')
        group.sio.sleep(2) # remove after testing
        # TODO 1: Collect all the parameters from group.client_data
        print('received data: ', group.client_data)
        # Example: model[0] = bytes_to_model(group.client_data[0]['model'])
        # TODO 2: Check the necessity of clustering
        split = False
        clients = []
        new_group = None
        # TODO 3: If necessary, call server.split_group(group, clients) to split the group
        if split:
            new_group = server.split_group(group, clients)
        # TODO 4: Average the parameters and update the model
        data = "new model"
        # 5: Convert data to bytes and start next round of training
        group.start_train(data)
        if new_group is not None:
            new_group.start_train(data)

    def create_data_callback():
        """
        When we want to launch a group with no initial data, this function is called.
        The initial model may change according to the requirements.
        """
        return "initial model"

    netserver = NetworkServer(group_complete_callback, create_data_callback)
    netserver.start(Host_Port, Host_IP)
    ##########################################

else:
    print("Error: Wrong type")
