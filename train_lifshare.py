import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import ssl

from archs.resnet.resnet_snn import ResNet19
from archs.resnet.resnet_snn_layershare import ResNet19_layershare # layer wise share
from archs.resnet.resnet_snn_channelshare import ResNet19_channelshare # channel wise share
from archs.resnet.resnet_snn_layerchannelshare import ResNet19_layerchannelshare # blockchannel wise share

from archs.vgg.vgg_snn import VGG16
from archs.vgg.vgg_layershare import VGG16_layershare  # layer wise share
from archs.vgg.vgg_channelshare import VGG16_channelshare
from archs.vgg.vgg_layerchannelshare import VGG16_layerchannelshare

from torch.utils.data import Dataset
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net
from torchvision import datasets

ssl._create_default_https_context = ssl._create_unverified_context

def main():
    args = config.get_args()

    # define dataset
    train_transform, valid_transform = data_transforms(args)

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 100


    # define model
    if args.arch == 'vgg16':
        if args.lifshare == 'noshare':
            model = VGG16(num_classes=n_class, total_timestep=args.timestep).cuda()
        elif args.lifshare == 'layer':
            model = VGG16_layershare(num_classes=n_class, total_timestep=args.timestep).cuda()
        elif args.lifshare == 'channel':
            model = VGG16_channelshare(num_classes=n_class, total_timestep=args.timestep, num_chgroup=args.ch_group_num).cuda()
        elif args.lifshare == 'layerchannel':
            model = VGG16_layerchannelshare(num_classes=n_class, total_timestep=args.timestep).cuda()
        else:
            print('no implementation')
            exit()
    elif args.arch == 'resnet19':
        if args.lifshare == 'noshare':
            model = ResNet19(args, num_classes=n_class, total_timestep=args.timestep).cuda()
        elif args.lifshare == 'layer':
            model = ResNet19_layershare(args,num_classes=n_class, total_timestep=args.timestep).cuda()
        elif args.lifshare == 'channel':
            model = ResNet19_channelshare(args,num_classes=n_class, total_timestep=args.timestep, num_chgroup=args.ch_group_num).cuda()
        elif args.lifshare == 'layerchannel':
            model = ResNet19_layerchannelshare(args,num_classes=n_class, total_timestep=args.timestep).cuda()
        else:
            print('no implementation')
            exit()




    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.epochs), eta_min= 0)


    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler)
        scheduler.step()
        if (epoch + 1) % args.valid_freq == 0:
            validate(args, epoch, val_loader, model, criterion)
            if args.lifshare == 'channel':
                utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1,
                                      tag=str(args.lifshare) + '_group' + str(args.ch_group_num) + '_D' + str(
                                          args.dataset) + '_A' + str(args.arch) + '_ce')
            elif args.lifshare == 'layerchannel':
                utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1,
                                      tag=str(args.lifshare) + '_group' + str(args.ch_group_num) + '_D' + str(
                                          args.dataset) + '_A' + str(args.arch) + '_ce')
            else:
                utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1,
                                      tag=str(args.lifshare) + '_D' + str(args.dataset) + '_A' + str(args.arch) + '_ce')

    utils.time_record(start)


def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()

    top1 = utils.AvgrageMeter()
    train_loss = 0.0

    if (epoch + 1) % 10 == 0:
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_lr()[0]))

    for step, (inputs, targets) in enumerate(train_data):

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        output_list = model(inputs)

        loss = criterion(sum(output_list)/ args.timestep, targets)
        loss.backward()

        prec1, prec5 = utils.accuracy(sum(output_list), targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()

        optimizer.step()
        reset_net(model)


    if (epoch + 1) % 10 == 0:
        print('train_loss: %.6f' % (train_loss / len(train_data)), 'train_acc: %.6f' % top1.avg)


def validate(args, epoch, val_data, model, criterion):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            loss = criterion(sum(outputs), targets)
            val_loss += loss.item()
            prec1, prec5 = utils.accuracy(sum(outputs), targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)

            reset_net(model)

        print('[Val_Accuracy epoch:%d] val_acc:%f'
              % (epoch + 1,  val_top1.avg))
        return val_top1.avg


if __name__ == '__main__':
    main()
