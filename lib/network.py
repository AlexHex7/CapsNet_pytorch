from lib.activation_function import squash
from lib.caps_conv import CapsConv
from lib.caps_fc import CapsFC
from torch import nn
import config as cfg
from lib.cuda_utils import variable


class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=cfg.channel, out_channels=256,
                               kernel_size=9, stride=1, padding=0,
                               bias=True)
        self.relu = nn.ReLU()

        self.primary_caps = CapsConv(in_channel=256, out_channel=32, vector_length=8)

        self.digit_caps = CapsFC(in_capsule=1152, in_length=8, out_capsule=10, out_length=16)

        self.decoder = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid(),
        )

    def forward(self, x, y=None, train=True):
        '''
        :param x: (b, 1, h, w)
        :param y: (b, 1)
        :return:  (b, 10, 16), (b, 1, 28, 28)
        '''

        batch_size = x.size(0)
        # ======= Conv1 =============
        # (b, 256, 20, 20)
        output = self.conv1(x)

        output = self.relu(output)

        # ======= PrimaryCaps =============
        # (b, 32, 6, 6, 8)
        output = self.primary_caps(output)
        vec_len = output.size(-1)
        # (b, 1152, 8)
        output = output.view(batch_size, -1, vec_len)
        output = squash(output, 2)

        # ======= Weight_Matrix =============
        # (b, 10, 16)
        predict = self.digit_caps(output)
        masked = []

        if not train:
            # (b, 10)
            length = torch.sqrt((predict**2).sum(-1))
            _, max_id = length.max(1)

            for batch_id in range(batch_size):
                a_sample = predict[batch_id]
                masked.append(a_sample[max_id[batch_id]])
            masked = torch.stack(masked, dim=0)

        else:
            for batch_id in range(batch_size):
                a_sample = predict[batch_id]
                masked.append(a_sample[y[batch_id]])
            masked = torch.stack(masked, dim=0)

        masked = masked.view(batch_size, -1)
        reconstruct_img = self.decoder(masked)
        reconstruct_img = reconstruct_img.view(batch_size, 1, 28, 28)

        assert predict.size() == (batch_size, 10, 16)
        assert reconstruct_img.size() == (batch_size, 1, 28, 28)

        return predict, reconstruct_img

    @staticmethod
    def margin_loss(predict, label, size_average=True):
        '''

        :param predict: (b, 10, 16)
        :param label:  (b, 1)
        :return:
        '''

        batch_size = predict.size(0)
        # (b, 10)
        label = variable(torch.zeros(batch_size, 10)).scatter_(1, label, 1)
        # (b, 10)
        v_dis = torch.sqrt((predict**2).sum(-1))
        zero = variable(torch.zeros(1))

        if torch.cuda.is_available():
            label = label.cuda()
            zero = zero.cuda()

        # (b, 10)
        max_l = torch.max(zero, cfg.m_plus - v_dis)
        max_r = torch.max(zero, v_dis - cfg.m_minus)
        T_c = label
        # (b, 10)
        L_c = (T_c * max_l**2) + (cfg.lambda_of_margin_loss * (1 - T_c) * max_r**2)


        L_c = L_c.sum()
        if size_average:
            L_c = L_c / batch_size
        return L_c

    @staticmethod
    def reconstruction_loss(img, reconstruction_img, size_average=True):
        '''

        :param img: (b, 1, 28, 28)
        :param reconstruction_img: (b, 1, 28, 28)
        :param size_average:
        :return:
        '''
        batch_size = img.size(0)

        loss = ((img - reconstruction_img)**2).sum() * 0.0005
        if size_average:
            loss = loss / batch_size
        return loss

    @staticmethod
    def calc_acc(predict, label):
        '''

        :param predict: (b, 10, 16)
        :param label: (b, 1)
        :return:
        '''

        batch_size = predict.size(0)
        _, predict_label = (predict**2).sum(-1).max(1, keepdim=True)
        acc = sum(predict_label.data == label.data)[0] / batch_size
        # print(sum(predict_label.cpu().data == label.cpu().data)[0], acc)
        return acc


if __name__ == '__main__':
    import torch
    from torch.autograd import Variable

    img = Variable(torch.randn(3, 1, 28, 28))
    y = Variable(torch.LongTensor([6, 3, 2])).view(3, 1)
    net = CapsNet()
    net(img, y)
