
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Decoder(nn.Module):
    def __init__(self, num_classes, output_unit_size, input_width,
                 input_height, num_conv_in_channel, cuda_enabled):
        super(Decoder, self).__init__()

        self.cuda_enabled = cuda_enabled

        fc1_output_size = 512
        fc2_output_size = 1024
        self.fc3_output_size = input_width * input_height * num_conv_in_channel
        self.fc1 = nn.Linear(num_classes * output_unit_size, fc1_output_size) # input dim 10 * 16.
        self.fc2 = nn.Linear(fc1_output_size, fc2_output_size)
        self.fc3 = nn.Linear(fc2_output_size, self.fc3_output_size)
        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, target):
        batch_size = target.size(0)
        masked_caps = utils.mask(x, self.cuda_enabled)
        vector_j = masked_caps.view(x.size(0), -1) # reshape the masked_caps tensor

        fc1_out = self.relu(self.fc1(vector_j))
        fc2_out = self.relu(self.fc2(fc1_out)) # shape: [batch_size, 1024]
        reconstruction = self.sigmoid(self.fc3(fc2_out)) # shape: [batch_size, fc3_output_size]

        assert reconstruction.size() == torch.Size([batch_size, self.fc3_output_size])

        return reconstruction
