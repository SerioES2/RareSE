import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import Model.CNN as cnn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameter
epochs = 5
output_classes = 10
batch_size = 100
learning_rate = 0.01

# Train the model
class Trainer():
  def __init__(self, input_data, model, cri, opt):
    self.__input_data = input_data
    self.__model = model
    self.__criterion = cri
    self.__optimizer = opt

  def Execute(self, epochs):
    total_step = len(self.__input_data)
    for epoch in range(epochs):
      for i, (images, labels) in enumerate(self.__input_data):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = self.__model(images)
        loss = self.__criterion(outputs, labels)

        # Backward
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

        if (i+1) % 100 == 0:
          print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, epochs, i+1, total_step, loss.item()))
    
  def SaveModel(self):
    torch.save(self.__model.state_dict(), './mnist.pth')

  def evaluate(self, eval_data):
    with torch.no_grad():
      correct = 0
      total = 0
      for images, labels, in eval_data:
        images = images.to(device)
        labels = labels.to(device)
        outputs = self.__model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

      print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def RunTrain(isSave=True):

  # MNIST dataset
  train_dataset = torchvision.datasets.MNIST(root='/data', train=True, transform=transforms.ToTensor(), download=True)
  test_dataset  = torchvision.datasets.MNIST(root='/data', train=False, transform=transforms.ToTensor())
  # Data loader
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

  # cnn model
  model = cnn.ConvNet(output_classes).to(device)
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

  # Training
  trainer = Trainer(train_loader, model=model, cri=criterion, opt=optimizer)
  trainer.Execute(epochs)

  if isSave == True:
    trainer.SaveModel()