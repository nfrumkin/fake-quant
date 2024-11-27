import torch

class BaseQuantizer(torch.nn.Module):
    def __init__(self, quantization_config):
        super(self).__init__()
        self.quantization_config = quantization_config

    def quant(self):
        raise NotImplementedError
    
    def dequant(self):
        raise NotImplementedError
    
    def forward(self, x):
        x = self.quant(x)
        return self.dequant(x)