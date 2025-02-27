# -*- coding: utf-8 -*-
# python main.py cifar10 --beta=0.01 --method ours --score_type ours
# python main.py cifar100 --beta=0.005 --method ours --score_type ours

import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn import WideResNet

import utils.svhn_loader as svhn
from utils.display_results import get_measures, print_measures
from utils.tinyimages_80mn_loader import TinyImages
import torchvision

parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')

# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.07, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--oe_batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
# DAL hyper parameters
parser.add_argument('--gamma', default=10, type=float)
parser.add_argument('--beta',  default=0.01, type=float)
parser.add_argument('--rho',   default=10, type=float)
parser.add_argument('--strength', default=1.0, type=float)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--iter', default=10, type=int)
# Others
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=1, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
# Energy-OE hyper parameters
parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
# method and score function type
parser.add_argument('--method', type=str, default='oe', help='method: ours, dal, oe, energy-oe')
parser.add_argument('--score_type', type=str, default='msp', help='energy, ours')
parser.add_argument('--model_path', type=str, default='./reproduce/cifar10_ours.pt', help='path to a well-trained model')


args = parser.parse_args()
torch.manual_seed(1)
np.random.seed(args.seed)
torch.cuda.manual_seed(1)


cudnn.benchmark = True  # fire on all cylinders

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
test_transform_imagenet = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

data_path = './data'

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10(data_path, train=True, transform=train_transform)
    test_data = dset.CIFAR10(data_path, train=False, transform=test_transform)
    cifar_data = dset.CIFAR100(data_path, train=False, transform=test_transform)
    num_classes = 10
else:
    train_data_in = dset.CIFAR100(data_path, train=True, transform=train_transform)
    test_data = dset.CIFAR100(data_path, train=False, transform=test_transform)
    cifar_data = dset.CIFAR10(data_path, train=False, transform=test_transform)
    num_classes = 100

transform_for_ood = trn.Compose([trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4), trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)])
ood_x = np.load(data_path + '/300K_random_images.npy')
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform):
        self.data = data
        self.label = torch.from_numpy(label)
        self.transform = transform
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        data = self.transform(data)

        return data, label 

    def __len__(self):
        return self.data.shape[0]
ood_data = MyDataset(ood_x, np.zeros(ood_x.shape[0]), transform_for_ood)


train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)


texture_data = dset.ImageFolder(root=data_path + "/ood_data/dtd/images", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
svhn_data = svhn.SVHN(root=data_path +'/ood_data/svhn/', split="test",transform=trn.Compose( [trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
places365_data = dset.ImageFolder(root=data_path +"/ood_data/places365", transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunc_data = dset.ImageFolder(root=data_path +"/ood_data/LSUN", transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]))
lsunfix_data = dset.ImageFolder(root=data_path +"/ood_data/LSUN", transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
isun_data = dset.ImageFolder(root=data_path +"/ood_data/iSUN",transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
imagenet_data_resize = torchvision.datasets.ImageFolder(data_path + '/Imagenet_resize', test_transform)

texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
svhn_loader = torch.utils.data.DataLoader(svhn_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places365_loader = torch.utils.data.DataLoader(places365_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunc_loader = torch.utils.data.DataLoader(lsunc_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
lsunfix_loader = torch.utils.data.DataLoader(lsunfix_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
isun_loader = torch.utils.data.DataLoader(isun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
cifar_loader = torch.utils.data.DataLoader(cifar_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
imagenet_loader_resize = torch.utils.data.DataLoader(imagenet_data_resize, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, score_type='msp', in_dist=False):
    _score = []
    net.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output, emb = net.pred_emb(data)
            if score_type == 'msp':
                smax = to_np(F.softmax(output, dim=1))
                _score.append(-np.max(smax, axis=1))
            elif score_type == 'energy':
                temper = 1
                conf = temper * (torch.logsumexp(output / temper, dim=1))
                _score.append(-conf.data.cpu().numpy())
            elif score_type == 'ours':
                target = torch.argmax(output.data, 1).detach()
                emb = emb/torch.norm(emb, dim=1, keepdim=True)
                a = net.fc.weight.data/torch.norm(net.fc.weight.data, dim=1, keepdim=True)
                conf1 = torch.norm((emb @ a.T), p=1, dim=1)
                # _score.append(-conf1.cpu().detach().numpy())
                smax = to_np(F.softmax(output, dim=1))
                conf2 = -np.max(smax, axis=1)
                _score.append(-conf1.cpu().detach().numpy()+conf2)
    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()


def get_and_print_results(ood_loader, in_score, score_type='msp', num_to_avg=1):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, score_type)
        print(out_score.shape)
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    print_measures(auroc, aupr, fpr, '')
    return fpr, auroc, aupr


def test():
    net.eval()
    correct = 0
    y, c = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    return correct / len(test_loader.dataset) * 100


net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate).cuda()
net.load_state_dict(torch.load(args.model_path))

############################## Testing ###################################     
net.eval()
in_score = get_ood_scores(test_loader, score_type=args.score_type, in_dist=True)
metric_ll = []
metric_ll.append(get_and_print_results(svhn_loader, in_score, args.score_type))
metric_ll.append(get_and_print_results(lsunc_loader, in_score, args.score_type))
metric_ll.append(get_and_print_results(isun_loader, in_score, args.score_type))
metric_ll.append(get_and_print_results(texture_loader, in_score, args.score_type))
metric_ll.append(get_and_print_results(places365_loader, in_score, args.score_type))
print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))

