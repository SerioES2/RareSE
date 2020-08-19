from abc import abstractmethod
import torch

'''
Description
--------------
    Mnist CNN分類モデルの学習処理実行クラス
'''
class Trainer():
  '''
  Description
  --------------
      コンストラクタ
  Parameter
  --------------
      - train_loader : 学習用データローダ
      - model : 学習用モデル
      - cri : 誤差関数
      - opt : 最適化関数
  '''
  def __init__(self, train_loader, model, cri, opt, device):
    self.train_loader = train_loader 
    self.model = model
    self.criterion = cri
    self.optimizer = opt
    self.device = device

  @abstractmethod
  def Execute(self):
    pass

  '''
  Description
  --------------
      学習後モデルデータを保存する
  ''' 
  def SaveModel(self):
    torch.save(self.model.state_dict(), './mnist.pth')

  '''
  Description
  --------------
      学習後モデルを取得
  ''' 
  def GetModel(self):
    return self.model

class MLPTrainer(Trainer):
  def __init__(self, train_loader, model, cri, opt, device):
    super(MLPTrainer, self).__init__(train_loader, model, cri, opt, device)

  '''
  Description
  --------------
      コンストラクタ
  Parameter
  --------------
      - epochs : エポック数
  Return
  --------------
      - 学習結果
  '''
  def Execute(self, epochs):

    train_result = {}

    for epoch in range(epochs):
      print('>> epoch : ', str(epoch))
      batch_loss = 0
      batch_accuracy = 0
      total = 0
      for i, (images, labels) in enumerate(self.train_loader):
        images = images.reshape(-1, 28*28).to(self.device)
        labels = labels.to(self.device)

        # Forward
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # training process
        _, prediction = torch.max(outputs.data, 1)
        batch_loss += loss.item()
        batch_accuracy += (prediction == labels).sum().item()
        total += labels.size(0)

      train_loss = batch_loss / len(self.train_loader)
      train_accuracy = batch_accuracy / total 

      train_result[epoch] = {'loss' : train_loss, 'accuracy' : train_accuracy}

      print('-----------------------------------------------')
      print('Epoch {} loss : {} accuracy : {}'.format(epoch, train_loss, train_accuracy ) )
      print('-----------------------------------------------')

    return train_result




class CNNTrainer(Trainer):
  def __init__(self, train_loader, model, cri, opt, device):
    super(CNNTrainer, self).__init__(train_loader, model, cri, opt, device)

  '''
  Description
  --------------
      コンストラクタ
  Parameter
  --------------
      - epochs : エポック数
  Return
  --------------
      - 学習結果
  '''
  def Execute(self, epochs):
    batch_loss = 0
    batch_accuracy = 0
    total = 0

    train_result = {}

    for epoch in range(epochs):
      print('>> epoch : ', str(epoch))
      for i, (images, labels) in enumerate(self.__train_loader):
        images = images.to(self.__device)
        labels = labels.to(self.__device)

        # Forward
        outputs = self.__model(images)
        loss = self.__criterion(outputs, labels)

        # Backward
        self.__optimizer.zero_grad()
        loss.backward()
        self.__optimizer.step()

        # training process
        _, prediction = torch.max(outputs.data, 1)
        batch_loss += loss.item()
        batch_accuracy += (prediction == labels).sum().item()
        total += labels.size(0)

      train_loss = batch_loss / len(self.__train_loader)
      train_accuracy = batch_accuracy / total 

      train_result[epoch] = {'loss' : train_loss, 'accuracy' : train_accuracy}

      print('-----------------------------------------------')
      print('Epoch {} loss : {} accuracy : {}'.format(epoch, train_loss, train_accuracy ) )
      print('-----------------------------------------------')

    return train_result

