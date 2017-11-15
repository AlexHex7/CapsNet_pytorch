import logging

import torch
import torch.utils.data as Data
import torchvision
import torchvision.utils as vutils
from lib.network import CapsNet

import config as cfg
from lib.cuda_utils import variable

logging.getLogger().setLevel(logging.INFO)

train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST(root='./mnist/',
                                       transform=torchvision.transforms.ToTensor(),
                                       train=False)
train_loader = Data.DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=cfg.batch_size, shuffle=False)

train_batch_num = len(train_loader)
test_batch_num = len(test_loader)

net = CapsNet()
opt = torch.optim.Adam(net.parameters(), lr=cfg.LR)

if torch.cuda.is_available():
    net.cuda()

if cfg.load_model:
    net.load_state_dict(torch.load(cfg.model_path))

for epoch_index in range(cfg.epoch):
    for train_batch_index, (img_batch, label_batch) in enumerate(train_loader):
        img_batch = variable(img_batch)
        label_batch = variable(label_batch).unsqueeze(dim=1)

        predict, reconstruct_img = net(img_batch, label_batch, train=True)

        acc = net.calc_acc(predict, label_batch)
        margin_loss = net.margin_loss(predict, label_batch)
        reconstruct_loss = net.reconstruction_loss(img_batch, reconstruct_img)

        loss = margin_loss + reconstruct_loss
        net.zero_grad()
        loss.backward()
        opt.step()

        logging.info('epoch[%d/%d] batch[%d/%d] loss:%.4f marin_loss:%.4f re_loss:%.4f acc:%.4f' %
              (epoch_index, cfg.epoch, train_batch_index, train_batch_num, loss.data[0], margin_loss.data[0],
               reconstruct_loss.data[0], acc))

        if (train_batch_index + 1) % cfg.test_per_batch == 0:
            total_margin_loss = 0
            total_reconstruct_loss = 0
            total_acc = 0

            for test_batch_index, (img_batch, label_batch) in enumerate(test_loader):
                img_batch = variable(img_batch, volatile=True)
                label_batch = variable(label_batch, volatile=True).unsqueeze(dim=1)

                predict, reconstruct_img = net(img_batch, train=False)

                acc = net.calc_acc(predict, label_batch)
                margin_loss = net.margin_loss(predict, label_batch)
                reconstruct_loss = net.reconstruction_loss(img_batch, reconstruct_img)

                total_margin_loss += margin_loss
                total_reconstruct_loss += reconstruct_loss
                total_acc += acc

                if test_batch_index == 0:
                    if torch.cuda.is_available():
                        img_batch = img_batch.cpu().data[:128]
                        reconstruct_img = reconstruct_img.cpu().data[:128]
                    else:
                        img_batch = img_batch.data[:128]
                        reconstruct_img = reconstruct_img.data[:128]
                    vutils.save_image(img_batch, 'result_img/input_img.png', nrow=16)
                    vutils.save_image(reconstruct_img, 'result_img/reconstruct_img_epoch%d_batch%d.png'
                                      % (epoch_index, train_batch_index), nrow=16)

            mean_acc = total_acc / test_batch_num
            mean_margin_loss = total_margin_loss / test_batch_num
            mean_reconstruct_loss = total_reconstruct_loss / test_batch_num
            mean_loss = mean_margin_loss + mean_reconstruct_loss
            logging.info('Test==> acc:%.4f loss:%.4f margin_loss:%.4f, reconstruct_loss:%.4f' %
                  (mean_acc, mean_loss.data[0], mean_reconstruct_loss.data[0], mean_margin_loss.data[0]))

            torch.save(net.state_dict(), cfg.model_path)





