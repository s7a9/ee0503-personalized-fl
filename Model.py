import torch
import torch.nn.functional as F
import torch.nn as nn

# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(6, 12, 5)
#         self.fc1 = torch.nn.Linear(12 * 5 * 5, 100)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 12 * 5 * 5)
#         x = self.fc1(x)
#         return x
    
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc1 = torch.nn.Linear(3 * 32 * 32, 100)
        

#     def forward(self, x):
#         x = F.relu(x)
#         x = x.view(-1, 3 * 32 * 32)
#         x = self.fc1(x)
#         return x

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = []

        # TODO: Build your own layers here!!!
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(3, 32, kernel_size=3, padding=1))
        self.layers.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.layers.append(nn.Conv2d(64, 64, kernel_size=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(64))

        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.AvgPool2d(1, stride=2))
        self.layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        self.layers.append(nn.Conv2d(128, 128, kernel_size=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(128))

        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.AvgPool2d(1, stride=2))
        self.layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        self.layers.append(nn.Conv2d(256, 256, kernel_size=1))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.BatchNorm2d(256))

        self.layers.append(nn.Flatten())
        
        num_features = 256 * 8 * 8

        self.layers.append(nn.Linear(num_features, 100))
        self.layers.append(nn.Softmax(dim=-1))
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x