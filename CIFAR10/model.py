import torch
from torch import nn

class GoogleNet(nn.Model):
    def __init__(self, input_size, output_size):
        super.__init__()

        self.model = nn.Sequential(
            GoogleNetInception()
        )

    
    def forward():
        return 0
    
class GoogleNetInception(nn.Module):
    def __init__(self, input_size, output_size):
        super.init()

        self.model = nn.Sequential(
            
        )