import torch

class GradExtractor(object):
    """
    For saving gradient for Grad-CAM
    """
    def __init__(self):
        pass

    def store_grad(self, grad):
        self.grad = grad

    def forward(self, module, x, target_name):
        # Only when None
        if target_name == None:
            if x.requires_grad == False:
                x.requires_grad = True
            self.feature = x
            x.register_hook(self.store_grad)
        
        for name, m in module._modules.items():
            x = m(x)
            if (target_name is not None) and (name == target_name):
                x.register_hook(self.store_grad)
                self.feature = x
        return x


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


def fix_module(module):
    """ Set all "requires_grad" of parameters in the module to False """
    for parameter in module.parameters():
        parameter.requires_grad = False
