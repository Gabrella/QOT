import argparse
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
from models import resnet_v1 as resnet

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./datasets/RAFDB/')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N')
parser.add_argument('--gpu', default='4', type=str)
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0

    print('Training time: ' + now.strftime("%m-%d %H:%M"))

    # create model
    model_cla = resnet.resnet50()
    model_cla = torch.nn.DataParallel(model_cla).cuda()
    checkpoint = torch.load('./checkpoint_cnn/RAFDB/[09-15]-[14-45]-model_best.pth.tar')
    pre_trained_dict = checkpoint['state_dict']
    for k, v in pre_trained_dict.items():
        print(k, v.shape)
    model_cla.load_state_dict(pre_trained_dict)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    
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
    model_cla.eval()
    feature_1 = []
    feature_2 = []
    feature_3 = []
    label = []
    top1 = AverageMeter('Accuracy', ':6.3f')
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # compute output
            x_1, x_2, x_3, x_fc1, x_fc2, x_fc3, output = model_cla(images)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            x_1 = x_1.permute(0, 2, 3, 1)
            x_2 = x_2.permute(0, 2, 3, 1)
            x_3 = x_3.permute(0, 2, 3, 1)
            if i == 0:
                feature_1 = x_1.cpu().numpy()
                feature_2 = x_2.cpu().numpy()
                feature_3 = x_3.cpu().numpy()
                label = target.cpu().numpy()
            else:
                feature_1 = np.concatenate((feature_1, x_1.cpu().numpy()),axis=0)
                feature_2 = np.concatenate((feature_2, x_2.cpu().numpy()),axis=0)
                feature_3 = np.concatenate((feature_3, x_3.cpu().numpy()),axis=0)
                label = np.concatenate((label, target.cpu().numpy()),axis=0)

        print(' *** Accuracy {top1.avg:.3f}  *** '.format(top1=top1))
    # train
    # np.save("./orthognal_npy/train_1_RAFDB2_v1.npy",feature_1)
    # np.save("./orthognal_npy/train_2_RAFDB2_v1.npy",feature_2)
    # np.save("./orthognal_npy/train_3_RAFDB2_v1.npy",feature_3)
    # np.save("./orthognal_npy/train_label_RAFDB2_v1.npy",label)
    # # test
    np.save("./orthognal_npy/test_1_RAFDB2_v1.npy",feature_1)
    np.save("./orthognal_npy/test_2_RAFDB2_v1.npy",feature_2)
    np.save("./orthognal_npy/test_3_RAFDB2_v1.npy",feature_3)
    np.save("./orthognal_npy/test_label_RAFDB2_v1.npy",label)


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

if __name__ == '__main__':
    main()
