import torch
import torch.nn as nn


class SRCNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.seq = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, padding=2, padding_mode='replicate'),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=2, padding_mode='replicate'),
        nn.ReLU(),
        nn.Conv2d(32,3,kernel_size=5,padding=2,padding_mode='replicate')
    )

  def forward(self, X):
    X = self.seq(X)
    
    return X