import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def train_op(model, loader, optimizer, epochs=1):
    model.train()  
    for ep in range(epochs):
        running_loss, samples = 0.0, 0
        for x, y in tqdm(loader, desc=f"Epoch {ep+1}/{epochs}"): 
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = torch.nn.CrossEntropyLoss()(model(x), y)
            running_loss += loss.item()*y.shape[0]
            samples += y.shape[0]

            loss.backward()
            optimizer.step()  

        print(f"Epoch {ep+1}/{epochs}, Loss: {running_loss / samples:.4f}")
    return running_loss / samples
      
def eval_op(model, loader):
    model.train()
    samples, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.data, 1)

            samples += y.shape[0]
            correct += (predicted == y).sum().item()

    return correct/samples
