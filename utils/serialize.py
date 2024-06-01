import io
import torch
from Model import Model
from typing import Tuple
from ServerClient import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def client_to_bytes(client: Client) -> bytes:
    buffer = io.BytesIO()
    data = {
        'model_state': client.model.state_dict(),
        'optimizer_state': client.optimizer.state_dict(),
        'W': client.W,
        'dW': client.dW,
        'W_old': client.W_old,
        'id': client.id
    }
    torch.save(data, buffer)
    return buffer.getvalue()

def bytes_to_client(bytes_: bytes, optimizer_fn, data_set, data_loader, idnum) -> Client:
    buffer = io.BytesIO(bytes_)
    data = torch.load(buffer)
    # for key, value in data.items():
    #     print(key)
    client = Client(Model, optimizer_fn, data_set, data_loader, idnum)
    
    client.model.load_state_dict(data['model_state'])
    client.optimizer.load_state_dict(data['optimizer_state'])
    
    client.W = data['W']
    client.dW = data['dW']
    client.W_old = data['W_old']
    client.id = data['id']
    
    return client


