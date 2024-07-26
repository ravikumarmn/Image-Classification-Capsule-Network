import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils


class CapsuleLayer(nn.Module):
    def __init__(self, in_unit, in_channel, num_unit, unit_size, use_routing,
                 num_routing, cuda_enabled):
        super(CapsuleLayer, self).__init__()

        self.in_unit = in_unit
        self.in_channel = in_channel
        self.num_unit = num_unit
        self.use_routing = use_routing
        self.num_routing = num_routing
        self.cuda_enabled = cuda_enabled

        if self.use_routing:
            self.weight = nn.Parameter(torch.randn(1, in_channel, num_unit, unit_size, in_unit))
        else:
            self.conv_units = nn.ModuleList([
                nn.Conv2d(self.in_channel, 32, 9, 2) for u in range(self.num_unit)
            ])

    def forward(self, x):
        if self.use_routing:
            # Currently used by DigitCaps layer.
            return self.routing(x)
        else:
            # Currently used by PrimaryCaps layer.
            return self.no_routing(x)

    def routing(self, x):
        batch_size = x.size(0)

        x = x.transpose(1, 2) # dim 1 and dim 2 are swapped. out tensor shape: [128, 1152, 8]
        x = torch.stack([x] * self.num_unit, dim=2).unsqueeze(4)
        batch_weight = torch.cat([self.weight] * batch_size, dim=0)
        u_hat = torch.matmul(batch_weight, x)
        b_ij = Variable(torch.zeros(1, self.in_channel, self.num_unit, 1))
        if self.cuda_enabled:
            b_ij = b_ij.cuda()
        num_iterations = self.num_routing

        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)  # Convert routing logits (b_ij) to softmax.
            # c_ij shape from: [128, 1152, 10, 1] to: [128, 1152, 10, 1, 1]
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = utils.squash(s_j, dim=3)
            v_j1 = torch.cat([v_j] * self.in_channel, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1) # shape: [128, 10, 16, 1]

    def no_routing(self, x):
        unit = [self.conv_units[i](x) for i, l in enumerate(self.conv_units)]
        unit = torch.stack(unit, dim=1)
        batch_size = x.size(0)
        unit = unit.view(batch_size, self.num_unit, -1)
        return utils.squash(unit, dim=2) # dim 2 is the third dim (1152D array) in our tensor
