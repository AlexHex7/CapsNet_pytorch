import torch
from torch.autograd import Variable


def variable(tensor, volatile=False):
    if torch.cuda.is_available():
        return Variable(tensor, volatile=volatile).cuda()

    return Variable(tensor, volatile=volatile)
