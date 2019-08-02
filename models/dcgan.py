import torch
import torch.nn as nn
import torch.nn.functional as F


class DCGAN_G(nn.Module):
    def __init__(self):
        super(DCGAN_G, self).__init__()
        self.ngf = 128
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, self.ngf*8, 5, 2, bias=False),
            nn.BatchNorm2d(self.ngf*8),
            nn.LeakyReLU()
        )
