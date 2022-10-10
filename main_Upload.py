import argparse
from locale import normalize
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
import random
import numbers
import torch.nn.functional as F
from models import resnet_v1 as resnet

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./datasets/RAFDB_Face/')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_cnn/RAFDB/' + time_str + 'model.pth.tar')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint_cnn/RAFDB/'+time_str+'model_best.pth.tar')
parser.add_argument('--log_path', type=str, default='./log/RAFDB/' + time_str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', dest='lr')
parser.add_argument('--factor', default=0.1, type=float, metavar='FT')
# 30 is the best
parser.add_argument('--af', '--adjust-freq', default=20, type=int, metavar='N', help='adjust learning rate frequency')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--evaluate_path', type=str, default='' + time_str + 'model.pth.tar')
parser.add_argument('-e', '--evaluate', default=False, action='store_true', help='evaluate model on test set')
parser.add_argument('--gpu', default='4', type=str)
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model_cla = resnet.resnet50()
    model_cla.fc = nn.Linear(2048, 12666)
    model_cla = torch.nn.DataParallel(model_cla).cuda()
    # pretrianed on msceleb
    checkpoint = torch.load('./pretrained/resnet50_pretrained_on_msceleb.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    model_dict = model_cla.state_dict()
    for k, v in pre_trained_dict.items():
        if k in model_dict:
            print(k, v.shape)
    pretrained_dict = {k: v for k, v in pre_trained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_cla.load_state_dict(model_dict)
    model_cla.module.fc = nn.Linear(256*3, 7).cuda()

    # define loss function (criterion) and optimizer
    criterion_val = nn.CrossEntropyLoss().cuda()
    criterion_train =  nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model_cla.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay
                                )

    recorder = RecorderMeter(args.epochs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            best_acc = best_acc.to()
            model_cla.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    # RAF-DB
    normalize = transforms.Normalize(mean=[0.5758095, 0.4500876, 0.40176094],
                                      std=[0.20888616, 0.19142343, 0.18289249])

    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.Resize((224, 224)),
                                                             transforms.RandomHorizontalFlip(),
                                                             transforms.ToTensor(),
                                                             normalize]))

    test_dataset = datasets.ImageFolder(valdir,
                                        transforms.Compose([transforms.Resize((224, 224)),
                                                            transforms.ToTensor(),
                                                            normalize]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate:
        checkpoint = torch.load(args.evaluate_path)
        model_cla.load_state_dict(checkpoint['state_dict'])
        validate(val_loader, model_cla, criterion_val, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # update learning rate
        current_learning_rate = adjust_learning_rate(optimizer, epoch, args)
        print('Current learning rate: ', current_learning_rate)
        txt_name = args.log_path + '-log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        # train for one epoch
        train_acc, train_los = train(train_loader, model_cla, criterion_train, optimizer, epoch, args)

        # evaluate on validation set
        val_acc, val_los = validate(val_loader, model_cla, criterion_val, args)

        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        curve_name = args.log_path + '-log.png'
        recorder.plot_curve(curve_name)

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        print('Current best accuracy: ', best_acc.item())
        txt_name = args.log_path + '-log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

        save_checkpoint({'state_dict': model_cla.state_dict()}, is_best, args)
        end_time = time.time()
        epoch_time = end_time - start_time
        print("An Epoch Time: ", epoch_time)
        txt_name = args.log_path + '-log.txt'
        with open(txt_name, 'a') as f:
            f.write(str(epoch_time) + '\n')
        # scheduler.step(val_acc)

def train(train_loader, model_cla, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))
    soft_max = nn.Softmax(dim=1)

    # switch mode
    model_cla.train()

    for i, (images, target) in enumerate(train_loader):

        # length = len(images)
        images = images.cuda()
        target = target.cuda()
        # compute output
        x_1, x_2, x_3, x_fc1, x_fc2, x_fc3, output = model_cla(images)
        x_fc1 = soft_max(x_fc1)
        x_fc2 = soft_max(x_fc2)
        x_fc3 = soft_max(x_fc3)
        # compute loss
        loss_softmax = criterion(output, target)
        loss_orthognal = orthognal_loss(x_fc1, x_fc2, x_fc3)
        loss = loss_softmax + 0.2 * loss_orthognal

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i, args)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    soft_max = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            x_1, x_2, x_3, x_fc1, x_fc2, x_fc3, output = model(images)
            loss_softmax = criterion(output, target)
            x_fc1 = soft_max(x_fc1)
            x_fc2 = soft_max(x_fc2)
            x_fc3 = soft_max(x_fc3)
            loss_orthognal = orthognal_loss(x_fc1, x_fc2, x_fc3)
            loss = loss_softmax + 0.2 * loss_orthognal

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i, args)

        print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
        with open(args.log_path + '-log.txt', 'a') as f:
            f.write(' * Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)


def orthognal_loss(x, y, z):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    l_12 = torch.sum(x*y, dim=1)
    l_13 = torch.sum(x*z, dim=1)
    l_23 = torch.sum(y*z, dim=1)
    return torch.mean((l_12+l_13+l_23)/3, dim=-1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, args):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = args.log_path + '-log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (args.factor ** (epoch // args.af))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            # .contiguous让地址连续
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
