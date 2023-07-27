import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

# Training on Different ResNet Features

# For (256, 56, 56)
class LeNet_1(nn.Module):
    def __init__(self, in_shape, out_dim):
        super(LeNet_1, self).__init__()

        self.conv1 = nn.Conv2d(in_shape[0], in_shape[0], 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_shape[0], in_shape[0], 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_shape[0], in_shape[0], 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(6400, out_dim)

    def forward(self, in_features):
        x = self.pool1(F.relu(self.conv1(in_features)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 6400)
        return self.fc1(x)


# For (512, 28, 28)
class LeNet_2(nn.Module):
    def __init__(self, in_shape, out_dim):
        super(LeNet_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.BatchNorm2d(
                in_shape[0],
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.BatchNorm2d(
                in_shape[0],
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(12800, out_dim)

    def forward(self, in_features):
        x = self.pool1((self.conv1(in_features)))
        x = self.pool2((self.conv2(x)))
        x = x.view(-1, 12800)
        return self.fc1(x)


# For (1024, 14, 14)
class LeNet_3(nn.Module):
    def __init__(self, in_shape, out_dim):
        super(LeNet_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.BatchNorm2d(
                in_shape[0],
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0], in_shape[0], 3),
            nn.BatchNorm2d(
                in_shape[0],
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.Dropout(0.5),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4096, out_dim)

    def forward(self, in_features):
        x = self.pool1(self.conv1(in_features))
        x = self.pool2(self.conv2(x))
        x = x.view(-1, 4096)
        return self.fc1(x)


# For (2048, 7, 7)
class LeNet_4(nn.Module):
    def __init__(self, in_shape, out_dim):
        super(LeNet_4, self).__init__()
        self.conv1 = nn.Conv2d(in_shape[0], in_shape[0], 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8192, out_dim)

    def forward(self, in_features):
        x = self.pool1(F.relu(self.conv1(in_features)))
        x = x.view(-1, 8192)
        return self.fc1(x)


