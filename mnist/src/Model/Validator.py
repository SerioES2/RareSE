import torch


'''
Description
------------
    Mnist CNN分類モデルの評価を行うクラス
'''
class Validator():

    def __init__(self, valid_loader, model, criterion, device):
        self.__valid_loader = valid_loader
        self.__model = model
        self.__criterion = criterion
        self.__device = device
        return

    def Validate(self):
        
        total = 0
        batch_acc = 0
        batch_loss = 0

        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.__valid_loader):
                images = images.to(self.__device)
                labels = labels.to(self.__device)

                # forward
                outputs = self.__model(images)

                # calculate loss
                loss = self.__criterion(outputs, labels)
                batch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                batch_acc += (predicted == labels).sum().item()

        valid_loss = batch_loss / len(self.__valid_loader)
        valid_acc = batch_acc / total

        print('-----------------------------------------------')
        print('Validataion loss : {} accuracy : {}'.format(valid_loss, valid_acc) )
        print('-----------------------------------------------')

        return {'loss': valid_loss, 'accuracy' :valid_acc}

'''
Description
------------
    Mnist MLP分類モデルの評価を行うクラス
'''
class MLPValidator():

    def __init__(self, valid_loader, model, criterion, device):
        self.__valid_loader = valid_loader
        self.__model = model
        self.__criterion = criterion
        self.__device = device
        return

    def Validate(self):
        
        total = 0
        batch_acc = 0
        batch_loss = 0

        with torch.no_grad():
            for idx, (images, labels) in enumerate(self.__valid_loader):
                images = images.reshape(-1,28*28).to(self.__device)
                labels = labels.to(self.__device)

                # forward
                outputs = self.__model(images)

                # calculate loss
                loss = self.__criterion(outputs, labels)
                batch_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                batch_acc += (predicted == labels).sum().item()

        valid_loss = batch_loss / len(self.__valid_loader)
        valid_acc = batch_acc / total

        print('-----------------------------------------------')
        print('Validataion loss : {} accuracy : {}'.format(valid_loss, valid_acc) )
        print('-----------------------------------------------')

        return {'loss': valid_loss, 'accuracy' :valid_acc}