import torch

class QLinear(torch.nn.Linear):
    def __init__(self, module):
        super().__init__(module.in_features, module.out_features)
        self.weight = SymmetricQuantizer(module.weight.detach())
        self.bias = module.bias

    def quant(self):
        raise NotImplementedError
    
    def dequant(self):
        raise NotImplementedError