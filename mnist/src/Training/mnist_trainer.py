# general
import sys

# pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# sklearn
from sklearn.model_selection import KFold

# my library
sys.path.append('..')
import Model.CNN as cnn
import Model.MLP as mlp
import Model.Trainer as trainer
import Model.Validator as validator

import mlflow

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameter
epochs = 5
output_classes = 10
batch_size = 100
learning_rate = 0.01

def testRun():
  # MNIST dataset
  train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)
  
  cv_splits = 3
  kfold = KFold(n_splits = cv_splits, shuffle=True, random_state = 0)
  for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_dataset)):
    print(">> CV fold step ", str(fold_idx))
    # cnn model
    model = mlp.SimpleMLP(output_classes).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # Data loader
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)

    # Training
    mnist_trainer = trainer.MLPTrainer(train_loader, model=model, cri=criterion, opt=optimizer, device=device)
    train_result = mnist_trainer.Execute(epochs)



def RunTorchCV():

  with mlflow.start_run():

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)

    train_results = {}
    valid_results = {}

    cv_splits = 3
    kfold = KFold(n_splits = cv_splits, shuffle=True, random_state = 0)
    for fold_idx, (train_idx, valid_idx) in enumerate(kfold.split(train_dataset)):
      print(">> CV fold step ", str(fold_idx))
      # cnn model
      model = mlp.SimpleMLP(output_classes).to(device)
      # Loss and optimizer
      criterion = nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

      # Data loader
      train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True)
      valid_loader = DataLoader(Subset(train_dataset, valid_idx), batch_size=batch_size, shuffle=False)

      # Training
      mnist_trainer = trainer.MLPTrainer(train_loader, model=model, cri=criterion, opt=optimizer, device=device)
      train_result = mnist_trainer.Execute(epochs)
      trained_model = mnist_trainer.GetModel()
      train_results[fold_idx] = train_result

      # Validation
      mnist_validator = validator.MLPValidator(valid_loader, model=trained_model, criterion=criterion, device=device)
      valid_result = mnist_validator.Validate()
      valid_results[fold_idx] = valid_result

    mlflow.log_param("method_name", mlp.SimpleMLP(output_classes).__class__.__name__)
    mlflow.log_param("output_class", output_classes)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)

    mlflow.log_param("fold_type", kfold.__class__.__name__)
    mlflow.log_param("n_splits", cv_splits)
    mlflow.log_param("random_state", 0)

    mlflow.log_param("criterion", nn.CrossEntropyLoss.__class__.__name__)
    mlflow.log_param("optimizer", torch.optim.Adam.__name__)

    average_loss = 0
    average_acc = 0
    for fold_idx, cv_result in train_results.items():
      loss = cv_result[cv_splits-1]["loss"]
      acc  = cv_result[cv_splits-1]["accuracy"]
      average_loss += loss
      average_acc  += acc 
      mlflow.log_metric("fold_" + str(fold_idx) + "_loss", loss)
      mlflow.log_metric("fold_" + str(fold_idx) + "_accuracy", acc)

    average_loss = average_loss / cv_splits
    average_acc  = average_acc / cv_splits
    mlflow.log_metric("average_loss", average_loss)
    mlflow.log_metric("average_acc", average_acc)

  return valid_results

def RunTrain(isSave=False):

  # MNIST dataset
  train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)
  # Data loader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  # cnn model
  model = cnn.ConvNet(output_classes).to(device)
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

  # Training
  mnist_trainer = trainer.Trainer(train_loader, model=model, cri=criterion, opt=optimizer, device=device)
  mnist_trainer.Execute(epochs)
  trained_model = mnist_trainer.GetModel()

  if isSave == True:
    trainer.SaveModel()

  return trained_model

RunTorchCV()
#RunTrain(isSave=True)
