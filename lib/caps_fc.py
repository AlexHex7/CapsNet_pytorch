from lib.activation_function import squash
from torch import nn

import config as cfg
from lib.cuda_utils import variable
import torch

class CapsFC(nn.Module):
    '''
        ------------------------------------------------------
        in_capsule: number of input capsule (vector)
        in_length: length of input capsule
        out_capsule: number of output capsule (vector)
        out_length: length of output capsule
        ------------------------------------------------------
        input: (b, in_capsule, in_length)
        output: (b, out_capsule, out_length)
    '''
    def __init__(self, in_capsule, in_length, out_capsule, out_length):
        super(CapsFC, self).__init__()

        self.in_capsule = in_capsule
        self.in_length = in_length
        self.out_capsule = out_capsule
        self.out_length = out_length

        self.weight_matrix = nn.Parameter(torch.randn(1, in_capsule, out_capsule,
                                                      out_length, in_length),
                                          requires_grad=True)
        self.register_parameter('weight_matrix', self.weight_matrix)

    def forward(self, x):
        output = self.routing(x)

        return output

    def routing(self, x):
        '''
        :param x: (b, 1152, 8)
        :return:
        '''

        batch_size = x.size(0)
        # (b, 1152, 1, 8, 1)
        x = x.view(batch_size, self.in_capsule, 1, self.in_length, 1)
        # (b, 1152, 10, 8, 1)
        x = x.repeat(1, 1, 10, 1, 1)
        # (b, 1152, 10, 16, 8)
        # W = self.weight_matrix.repeat(1, 1, 1, 1, 1)
        # (b, 1152, 10, 16, 1)
        u_hat = torch.matmul(self.weight_matrix, x)

        # (b, 1152, 10, 1)
        b_ij = variable(torch.zeros(batch_size, self.in_capsule, self.out_capsule, 1))
        for iter_index in range(cfg.iteration+1):
            # (b, 1152, 10, 1)
            c_ij = nn.functional.softmax(b_ij)
            # (b, 1152, 10, 1, 1)
            c_ij = c_ij.unsqueeze(-1)
            # (b, 1152, 10, 16, 1)
            s_j = u_hat * c_ij
            # (b, 1, 10, 16, 1)
            s_j = s_j.sum(1, keepdim=True)
            v_j = squash(s_j, 3)
            # (b, 1152, 10, 16, 1)
            v_j_repeat = v_j.repeat(1, self.in_capsule, 1, 1, 1)
            # (b, 1152, 10, 1, 1)
            product = torch.matmul(u_hat.transpose(3, 4), v_j_repeat)
            # (b, 1152, 10, 1)
            product = product.squeeze(-1)
            b_ij = b_ij + product

        v_j = v_j.view(batch_size, self.out_capsule, self.out_length)
        # print(v_j.size())
        return v_j


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    x = Variable(torch.randn(3, 1152, 8))
    net = CapsFC(1152, 8, 10, 16)
    net(x)
