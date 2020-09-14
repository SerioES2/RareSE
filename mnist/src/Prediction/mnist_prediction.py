import os
import sys
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

sys.path.append('..')
from Model import CNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Predictor():
  def __init__(self):
    return

  def LoadModel(self, model_path):
    output_classes = 10
    self.__model = CNN.ConvNet(output_classes)
    self.__model.load_state_dict(torch.load(model_path, map_location=device))

  def ReadImage(self, image_file):
    image_data = Image.open(image_file)
    image_data = image_data.convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    image_data = transform(image_data).unsqueeze(0)
    return image_data

  def ConvertToTensor(self, image_data):
    out = image_data.convert('L')
    transform = transforms.Compose([transforms.ToTensor()])
    out = transform(out ).unsqueeze(0)
    return out

  def Predict(self, image_file):

    image_data = self.ReadImage(image_file)

    output = self.__model(image_data)
    _, result = torch.max(output, 1)
    return result

  def Predict2(self, image_data):
    tensor = self.ConvertToTensor(image_data)
    output = self.__model(tensor)
    _, result = torch.max(output, 1)
    return result