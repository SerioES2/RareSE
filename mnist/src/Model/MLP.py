import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#########
# MLP
# - 1 fully connected layer
class SimpleMLP(nn.Module):
  def __init__(self, classes = 10):
    super(SimpleMLP, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Linear(28 * 28, 100),
        nn.ReLU()
    )

    self.layer2 = nn.Sequential(
        nn.Linear(100, classes)
    )

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.softmax(out)
    return out

