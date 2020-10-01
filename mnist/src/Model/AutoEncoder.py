import torch
import torch.nn as nn

'''
AutoEncoder model : Neural network
    - Encoder : 2 full connected layer
        - 1 layer : 
    - Decoder : 2 full connected layer
'''
class AutoEncoder(nn.Module):
    
    self.__enc_shape1 = 128
    self.__enc_shape2 = 64
    self.__enc_shape3 = 32
    
    def __init__(self, input_shape=784):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, self.__enc_shape1),
            nn.ReLU(inplace=True),
            nn.Linear(self.__enc_shape1, self.__enc_shape2),
            nn.ReLU(inplace=True),
            nn.Linear(self.__enc_shape2, self.__enc_shape3)
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.__enc_shape3, self.__enc_shape2),
            nn.ReLU(inplace=True)
            nn.Linear(self.__enc_shape2, self.__enc_shape1),
            nn.ReLU(inplace=True)
            nn.Linear(self.__enc_shape1, input_shape)
        )
        
        return

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    

class ConvolutionalAutoEncoder(nn.Module):

    self.__conv_flatten_shape = 1568
    
    def __init__(self, image_size = 64):
        super(ConvolutionalAutoEncoder, self).__init__()
        # Convolutional layer
        self.encLayer1 = nn.Sequential(
            nn.Conv2d(1, image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(image_size, image_size*2, kernel_size+4, stride=2, padding=1),
            nn.BatchNorm2d(image_size*2),
            nn.ReLU(inplace=True)
        )
        
        # Full connected layer
        input_shape = 1568
        hidden_shape = int(input_shape/2)
        out_z_shape = int(input_shape/4)
        self.encLayer2 = nn.Sequential(
            nn.Linear(self.__conv_flatten_shape, hidden_shape),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_shape, out_z_shape),
            nn.ReLU(inplace=True)
        )
        
        # Full connected decoder layer
        self.decLayer1 = nn.Sequential(
            nn.Linear(out_z_shape, hidden_shape),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_shape, input_shape),
            nn.Sigmoid()
        )
        
        # Convolutional decoder layer
        self.decLayer2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=image_size*2, out_channels=image_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=image_size, out_channels=1, kernel_size, stride2, padding=1),
            nn.Sigmoid()
        )
    
    '''
    Forward process
    '''
    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    '''
    Encoder process
    '''
    def encoder(self, x):
        # convolution
        out = self.encLayer1(x)
        # full connected
        out = out.reshape(-1, self.__conv_flatten_shape)
        out = self.encLayer2(out)
        
    '''
    Decoder process
    '''
    def decoder(self, z):
        # full connected
        out = self.decLayer1(z)
        # convolution
        out = out.reshape(-1, 32, 7, 7)
        out = self.decLayer2(out)
        return out