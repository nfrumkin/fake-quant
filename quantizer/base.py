import torch

class BaseQuantizer(torch.nn.parameter.Parameter):
    def __init__(self, param):
        super().__init__()
        self.data = param.data
        self.requires_grad = param.requires_grad

    def quant(self):
        raise NotImplementedError
    
    def dequant(self):
        raise NotImplementedError
    
    # def forward(self, x):
    #     # x = self.quant(x)
    #     # return self.dequant(x)