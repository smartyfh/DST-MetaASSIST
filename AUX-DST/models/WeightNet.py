import torch
import torch.nn as nn


class WNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WNet, self).__init__()
        self.loss_to_weight = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(hidden_dim, output_dim),
                                            nn.Sigmoid()) 

    def forward(self, loss):
        return self.loss_to_weight(loss)
    
    
class WNet2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WNet2, self).__init__()
        self.loss_to_weight1 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_dim, output_dim),
                                             nn.Sigmoid()) 
        self.loss_to_weight2 = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(hidden_dim, output_dim),
                                             nn.Sigmoid()) 

    def forward(self, loss):
        return self.loss_to_weight1(loss), self.loss_to_weight2(loss)

    
class SlotWeight(nn.Module):
    def __init__(self, num_slots, init_val=0):
        super(SlotWeight, self).__init__()
        # if init_val == 0:
        #     self.weight = nn.Parameter(torch.zeros(1, num_slots, requires_grad=True))
        # else:
        self.weight = nn.Parameter(torch.ones(1, num_slots, requires_grad=True) * init_val)
        self.sigmoid = nn.Sigmoid()
                                   
    def forward(self):
        return self.sigmoid(self.weight)