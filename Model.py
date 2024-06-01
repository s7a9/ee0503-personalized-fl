
import torch
import torch.nn.functional as F

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
    
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(3 * 32 * 32, 100)
        

    def forward(self, x):
        x = F.relu(x)
        x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x