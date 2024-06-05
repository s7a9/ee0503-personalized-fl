import torch
import torch.nn.functional as F
import torch.nn as nn

# class ConvNet(torch.nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 8, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(8, 16, 5)
#         self.fc1 = torch.nn.Linear(16 * 5 * 5, 100)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = self.fc1(x)
#         return x
    
class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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



# import torch
# import torch.nn as nn

# class BasicBlock(nn.Module):
#     """Basic Block for resnet 18 and resnet 34

#     """

#     #BasicBlock and BottleNeck block
#     #have different output size
#     #we use class attribute expansion
#     #to distinct
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         #residual function
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#         )

#         #shortcut
#         self.shortcut = nn.Sequential()

#         #the shortcut output dimension is not the same with residual function
#         #use 1*1 convolution to match the dimension
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class BottleNeck(nn.Module):
#     """Residual block for resnet over 50 layers

#     """
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleNeck.expansion)
#             )

#     def forward(self, x):
#         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

# class ResNet(nn.Module):

#     def __init__(self, block, num_block, num_classes=100):
#         super().__init__()

#         self.in_channels = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True))
#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         """make resnet layers(by layer i didnt mean this 'layer' was the
#         same as a neuron netowork layer, ex. conv layer), one layer may
#         contain more than one residual block

#         Args:
#             block: block type, basic block or bottle neck block
#             out_channels: output depth channel number of this layer
#             num_blocks: how many blocks per layer
#             stride: the stride of the first block of this layer

#         Return:
#             return a resnet layer
#         """

#         # we have num_block blocks per layer, the first block
#         # could be 1 or 2, other blocks would always be 1
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         output = self.conv3_x(output)
#         output = self.conv4_x(output)
#         output = self.conv5_x(output)
#         output = self.avg_pool(output)
#         output = output.view(output.size(0), -1)
#         output = self.fc(output)

#         return output

# def ConvNet():
#     return ResNet(BasicBlock, [2, 2, 2, 2])