
### IMPORTS ###
import torch.nn as nn



### CLASS DEFINITION ###
class CustomELU(nn.Module):
    def __init__(self):
        super(CustomELU, self).__init__()

    def forward(self, x):
        ## Contractive and positive
        alpha = 0.9
        return alpha* nn.ELU()(x)+alpha