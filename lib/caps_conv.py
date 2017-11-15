from torch import nn
import torch


class CapsConv(nn.Module):
    '''
        has not implemented routing of CapsConv.
        ------------------------------------------------------
        in_channel: channel of input feature map.
        out_channel: channel of output feature map. Each channel contains one capsule.
        vector_length: conv_unit number (the number of kernels) in each capsule (channel).
        ------------------------------------------------------
        input: (b, in_channel, h, w)
        output: (b, out_channel, h, w, vec_len)
    '''
    def __init__(self, in_channel, out_channel, vector_length):
        super(CapsConv, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vector_length = vector_length

        self.layer = nn.Conv2d(in_channels=self.in_channel,
                               out_channels=self.out_channel * self.vector_length,
                               kernel_size=9,
                               stride=2,
                               bias=True)

    def forward(self, x):
        output = self.without_routing(x)
        assert output.size()[1] == self.out_channel
        assert output.size()[-1] == self.vector_length
        return output

    def without_routing(self, x):
        # (b, out_c*vec_len, h, w)
        output = self.layer(x)
        # [(b, vec_len, h, w), (b, vec_len, h, w), ...]
        conv_units = torch.split(output, 8, 1)
        # (b, out_c, vec_len, h, w)
        output = torch.stack(conv_units, dim=1)
        # (b, out_c, h, w, vec_len)
        output = output.permute(0, 1, 3, 4, 2).contiguous()
        return output


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.randn(3, 256, 20, 20))
    net = CapsConv(256, 32, 8)
    net(x)
