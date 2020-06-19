import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# CNN
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

def load_prediction_model(model_path):
  output_classes = 10
  model = ConvNet(output_classes)
  model.load_state_dict(torch.load(model_path, map_location=device))
  return model

def read_image(image_file):
  image_data = Image.open(image_file)
  image_data = image_data.convert('L')
  transform = transforms.Compose([transforms.ToTensor()])
  image_data = transform(image_data).unsqueeze(0)
  return image_data

def predict(model, image_data):
  output = model(image_data)
  _, result = torch.max(output, 1)
  return result