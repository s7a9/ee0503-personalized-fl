import argparse
import torch
import numpy as np
from Model import Model
from ServerClient import *
from Dataset import Cifar100Dataset
from network.server import NetworkServer
from network.client import run_client, send_data
from utils.serialize import *

parser = argparse.ArgumentParser(description='Input Operater')
parser.add_argument('--type', type=str, default="server") # or client
parser.add_argument('--ID', type=int, default=1)
parser.add_argument('--host', type=str, default="localhost")
parser.add_argument('-p', type=int, default="8192")
parser.add_argument("-k", type=int, default=2)
parser.add_argument("--round", type=int, default=5)
parser.add_argument('--noniid', type=str, default="off")
parser.add_argument('--clustering', type=str, default="off")
args = parser.parse_args()
ID = args.ID
Host_IP = args.host
Host_Port = args.p
Ptype = args.type
if args.clustering == "on":
    split = True
else:
    split = False

if args.noniid == "on":
    noniid = True
else:
    noniid = False


batch_size = 128
EPS_1 = 0.4
EPS_2 = 1.6

##########################################
# noniid分布处理 # 陈

# dataset
root="../data/cifar-100-python"
train_set = Cifar100Dataset(root=root, is_train=True, noniid = noniid) # 引入了noniid参数
val_set = Cifar100Dataset(root=root, is_train=False)
    
train_loader = DataLoader(train_set,
                                shuffle=True,
                                batch_size=batch_size)
val_loader = DataLoader(val_set,shuffle=False,batch_size=batch_size)

data_set = (train_set, val_set)
data_loader = (train_loader, val_loader)
    
##########################################

# optimizer = torch.optim.SGD(Model.parameters,lr=0.1, momentum=0.9)
optimizer = lambda x : torch.optim.Adam(x,lr=1e-4, betas=(0.9, 0.99))
idnum = ID

if Ptype == 'client':
    ##########################################
    # 传输参数设置
    def train_callback(sio, data): # receive data from server
        # TODO: Train the model, then send updated model to server
        print('=====================')
        print('Training model')
        
        if not isinstance(data, str):
            client = bytes_to_client(data, optimizer, data_set, data_loader, idnum)
        else:
            # optimizer = lambda x : torch.optim.Adam(x,lr=1e-4, betas=(0.9, 0.99))
            # idnum = ID
            client = Client(Model, optimizer, data_set, data_loader, idnum)
    
        client.compute_weight_update(epochs=1) # need for change
        print("test accuracy:" , client.evaluate())
        trained_data = client_to_bytes(client)
        
        sio.sleep(random.randint(1, 4)) # remove after testing
        print('Sending data')
        send_data(sio, trained_data)

    run_client(Host_Port, train_callback, Host_IP)
    ##########################################

elif Ptype == 'server':

    server = Server(Model, data_set, data_loader)
    ##########################################
    # 传输参数设置
    def group_complete_callback(netserver, group):
        """
        When all clients in a group have finished training, this function is called.
        """
        print('=====================')
        group.sio.sleep(2) # remove after testing
        # TODO 1: Collect all the parameters from group.client_data
        # print('received data: ', group.client_data)

        clients = []
        matchs = []
        for sids, datas in group.client_data.items():
            clients.append(bytes_to_client(datas, optimizer, data_set, data_loader, idnum))
            matchs.append(sids)

        # Example: model[0] = bytes_to_model(group.client_data[0]['model'])
        # TODO 2: Check the necessity of clustering
        # clients = []
        global split
        new_group = None
        cluster_indices = [np.arange(len(clients)).astype("int")]
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        # TODO 3: If necessary, call server.split_group(group, clients) to split the group
        if split:
            participating_clients = server.select_clients(clients, frac=1.0)
            similarities = server.compute_pairwise_similarities(clients)
            cluster_indices_new = []
            for idc in cluster_indices:
                max_norm = server.compute_max_update_norm([clients[i] for i in idc])
                mean_norm = server.compute_mean_update_norm([clients[i] for i in idc])
             
                # if mean_norm<EPS_1 and max_norm>EPS_2 and len(idc)>2:
                if True:
                    # server.cache_model(idc, clients[idc[0]].W, acc_clients)
            
                    c1, c2 = server.cluster_clients(similarities[idc][:,idc]) 
                    cluster_indices_new += [c1, c2]
                else:
                    cluster_indices_new += [idc]
        
        
            cluster_indices = cluster_indices_new
            client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]
            sid_clusters = [[matchs[i] for i in idcs] for idcs in cluster_indices]
            
            server.aggregate_clusterwise(client_clusters)

            acc_clients = [client.evaluate() for client in clients]
            # print(acc_clients)
            if len(client_clusters) == 2:
                new_group = netserver.split_group(group, sid_clusters[1])
            split = False


        # if split: # clients = [client1, client2, ...] is a subset of clients in original group to be split as a new group
        #     new_group = netserver.split_group(group, clients)

        # TODO 4: Average the parameters and update the model


        averaged_weights = server.average_client_weights(client_clusters[0])
        server.load_average_weights(averaged_weights)
        averaged_client =  Client(Model, optimizer, data_set, data_loader, idnum)
        data = averaged_client.weight_receive(averaged_weights)
        
        # 5: Convert data to bytes and start next round of training
        group.start_train(data)
        if len(client_clusters) == 1:
            print("test accuracy:" , averaged_client.evaluate())
        else:
            new_averaged_weights = server.average_client_weights(client_clusters[1])
            server.load_average_weights(new_averaged_weights)
            new_averaged_client =  Client(Model, optimizer, data_set, data_loader, idnum)
            print("test accuracy:" , averaged_client.evaluate(), " and ",new_averaged_client.evaluate())
            new_data = new_averaged_client.weight_receive(new_averaged_weights)
            new_group.start_train(new_data)

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
