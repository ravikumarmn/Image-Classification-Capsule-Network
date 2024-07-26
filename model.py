import torch
import torch.nn as nn
from torch.autograd import Variable

from conv_layer import ConvLayer
from capsule_layer import CapsuleLayer
from decoder import Decoder

class Net(nn.Module):
    def __init__(self, num_conv_in_channel, num_conv_out_channel, num_primary_unit,
                 primary_unit_size, num_classes, output_unit_size, num_routing,
                 use_reconstruction_loss, regularization_scale, input_width, input_height,
                 cuda_enabled):
        super(Net, self).__init__()

        self.cuda_enabled = cuda_enabled

        self.use_reconstruction_loss = use_reconstruction_loss
        self.image_width = input_width
        self.image_height = input_height
        self.image_channel = num_conv_in_channel
        self.regularization_scale = regularization_scale

        # Layer 1: Conventional Conv2d layer.
        self.conv1 = ConvLayer(in_channel=num_conv_in_channel,
                               out_channel=num_conv_out_channel,
                               kernel_size=9)
        # PrimaryCaps
        # Layer 2: Conv2D layer with `squash` activation.
        self.primary = CapsuleLayer(in_unit=0,
                                    in_channel=num_conv_out_channel,
                                    num_unit=num_primary_unit,
                                    unit_size=primary_unit_size, # capsule outputs
                                    use_routing=False,
                                    num_routing=num_routing,
                                    cuda_enabled=cuda_enabled)
        # DigitCaps
        # Final layer: Capsule layer where the routing algorithm is.
        self.digits = CapsuleLayer(in_unit=num_primary_unit,
                                   in_channel=primary_unit_size,
                                   num_unit=num_classes,
                                   unit_size=output_unit_size, # 16D capsule per digit class
                                   use_routing=True,
                                   num_routing=num_routing,
                                   cuda_enabled=cuda_enabled)
        # Reconstruction network
        if use_reconstruction_loss:
            self.decoder = Decoder(num_classes, output_unit_size, input_width,
                                   input_height, num_conv_in_channel, cuda_enabled)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_primary_caps = self.primary(out_conv1)
        out_digit_caps = self.digits(out_primary_caps)
        return out_digit_caps

    def loss(self, image, out_digit_caps, target, size_average=True):
        recon_loss = 0
        m_loss = self.margin_loss(out_digit_caps, target)
        if size_average:
            m_loss = m_loss.mean()

        total_loss = m_loss

        if self.use_reconstruction_loss:
            # Reconstruct the image from the Decoder network
            reconstruction = self.decoder(out_digit_caps, target)
            recon_loss = self.reconstruction_loss(reconstruction, image)

            # Mean squared error
            if size_average:
                recon_loss = recon_loss.mean()

            total_loss = m_loss + recon_loss * self.regularization_scale

        return total_loss, m_loss, (recon_loss * self.regularization_scale)

    def margin_loss(self, input, target):
        batch_size = input.size(0)

        # ||vc|| also known as norm.
        v_c = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms.
        zero = Variable(torch.zeros(1))
        if self.cuda_enabled:
            zero = zero.cuda()
        m_plus = 0.9
        m_minus = 0.1
        loss_lambda = 0.5
        max_left = torch.max(m_plus - v_c, zero).view(batch_size, -1)**2
        max_right = torch.max(v_c - m_minus, zero).view(batch_size, -1)**2
        t_c = target
        # Lc is margin loss for each digit of class c
        l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
        l_c = l_c.sum(dim=1)

        return l_c

    def reconstruction_loss(self, reconstruction, image):
        batch_size = image.size(0) # or another way recon_img.size(0)
        # error = (recon_img - image).view(batch_size, -1)
        image = image.view(batch_size, -1) # flatten 28x28 by reshaping to [batch_size, 784]
        error = reconstruction - image
        squared_error = error**2

        # Scalar Variable
        recon_error = torch.sum(squared_error, dim=1)

        return recon_error
