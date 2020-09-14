import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#########
# CNN 
# - 2 convolutional layer
#   - convolution1
#   - convolution2
# - 1 fully connected layer
class ConvNet(nn.Module):
  def __init__(self, classes = 10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.layer2 = nn.Sequential(
      nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.fc = nn.Linear(7*7*32, classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out

