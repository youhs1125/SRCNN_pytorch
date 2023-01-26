import torch
import torch.nn as nn

class SRCNN(nn.Module):
  def __init__(self):
    super().__init__()

    self.inputLayer = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, padding=2, padding_mode='replicate'),
        nn.ReLU()
    )
    self.midLayer = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=2, padding_mode='replicate'),
        nn.ReLU()
    )
    self.outputLayer = nn.Sequential(
        nn.Conv2d(32,3,kernel_size=5,padding=2,padding_mode='replicate'),
        nn.ReLU()
    )
  def forward(self, X):
    X = self.inputLayer(X)
    X = self.midLayer(X)
    X = self.outputLayer(X)
    
    return X