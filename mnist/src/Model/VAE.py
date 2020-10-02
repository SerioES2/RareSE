import torch
import torch.nn as nn
import torch.functional as F

class VAE(nn.Module):
    
    def __init__(self, in_shape=784):
        super(VAE, self).__init()
        
        self.__input_shape = 784
        self.__hidden_shape = 392
        self.__z_dim = 100

        # full connected encoder layer
        self.encLayer = nn.Sequential(
            nn.Linear(self.__input_shape, self.__hidden_shape),
            nn.ReLU(inplace=True)
        )
        
        # calculate the parameters of gausian distribution
        self.mean = nn.Linear(self.__hidden_shape, self.__z_dim)
        self.var  = nn.Linear(self.__hidden_shape, self.__z_dim)
        
        # full connected decoder layer
        self.decoder = nn.Sequential(
            nn.Linear(self.__z_dim, self.__hidden_shape),
            nn.ReLU(inplace=True),
            nn.Linear(self.__hidden_shape, self.__input_shape),
            nn.Sigmoid()
        )
        
    def encoder(self, x):
        out = self.encLayer(x)
        mean = self.mean(out)
        logvar = self.var(out)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mean + eps * std
    
    def decoder(self, z):
        out = self.decLayer(z)
        return out
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z_dim = self.reparameterize(mean, logvar)
        out = self.decoder(z_dim)
        return out, mean, logvar

    def loss(self, reconst_x, x, mu, logvar):
        reconst_loss = torch.mean(F.binary_cross_entropy(x))
        kl_penalty = 0.5 * sum(1 + logvar - mu**2 - torch.exp(logvar) )
        return reconst_loss + kl_penalty