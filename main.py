#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import ipdb
import shutil

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


from models.nin import NIN
from utils.averagevaluemeter import AverageValueMeter

parser = argparse.ArgumentParser(description='Network In Network Training')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=230, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', default=50, type=int, help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: None)')
parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-e', '--evaluate', action='store_true', help='evaluate model on validation set')


def main():
    global args
    args = parser.parse_args()
    torch.cuda.set_device(1)
    root = os.path.expanduser('~/datasets')
    train_dataset = CIFAR10(os.path.join(root, args.data), train=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    print(len(train_loader))

    test_dataset = CIFAR10(os.path.join(root, args.data), train=False,
                           transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args.workers)
    print(len(test_loader))

    model = NIN(num_classes=10)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()

    params = list()
    param_dict = dict(model.named_parameters())
    for key, value in param_dict.items():
        if key == 'features.20.weight':
            params.append({'params': [value], 'lr':0.1*args.lr, 'momentum':0.95, 'weight_decay':args.weight_decay})
        elif key == 'features.20.bias':
            params.append({'params': [value], 'lr':0.1*args.lr, 'momentum':0.95, 'weight_decay':0.0})
        elif 'weight' in key:
            params.append({'params':[value], 'lr':1.0*args.lr, 'momentum':0.95, 'weight_decay':args.weight_decay})
        else:
            params.append({'params':[value], 'lr':2.0*args.lr, 'momentum':0.95, 'weight_decay':0.0})
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

    best_acc = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_loader, model, criterion, optimizer, epoch, test_loader)

        acc = validate(test_loader, model, criterion)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({'epoch': epoch+1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer}, is_best)

def train(train_loader, model, criterion, optimizer, epoch, test_loader):
    losses = AverageValueMeter()

    model.train()

    for i, (input, label) in enumerate(train_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()

        input_var = Variable(input)
        label_var = Variable(label)
        output = model(input_var)

        loss = criterion(output, label_var)
        losses.update(loss.data[0], input.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  epoch, i, len(train_loader), loss=losses
                  ))

def validate(val_loader, model, criterion):
    accs = AverageValueMeter()
    losses = AverageValueMeter()
    model.eval()

    for i, (input, label) in enumerate(val_loader):
        if torch.cuda.is_available():
            input = input.cuda()
            label = label.cuda()
        input_var = Variable(input, volatile=True)
        label_var = Variable(label, volatile=True)

        output = model(input_var)
        loss = criterion(output, label_var)
        losses.update(loss.data[0], input.size(0))

        acc = accuracy(output, label)
        accs.update(acc, input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{}/{}]\t'
                  'Loss: {loss.val:.4f}({loss.avg:.4f})\t'
                  'Accuracy: {acc.val:.4f}({loss.avg:.4f})\t'.format(
                  i, len(val_loader), loss=losses, acc=accs
                  ))
    print('Accuracy {acc.avg:.4f}'.format(acc=accs))
    return accs.avg

def adjust_learning_rate(optimizer, epoch):
    if (epoch+1) % 30 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1

def accuracy(output, label):
    batch_size = label.size(0)
    _, pred = output.data.max(dim=1)
    acc = label.eq(pred).sum() / batch_size
    return acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == '__main__':
    main()
