# -*- coding: utf-8 -*-

import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F


import utils.svhn_loader as svhn
from utils.display_results import get_measures, print_measures
from utils.tinyimages_80mn_loader import TinyImages
import torchvision
import os
# import faiss
from visualize_utils import get_feas_by_hook

from torchvision import datasets, transforms
from torch.utils.data import Dataset
# from ImageNet-experiment.tiny-imagent-val import TinyImageNet_load
from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
from PIL import Image


parser = argparse.ArgumentParser(description='DAL training procedure on the CIFAR benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet-1k'],
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
parser.add_argument('--beta_dal',  default=0.01, type=float)
parser.add_argument('--rho',   default=10, type=float)
parser.add_argument('--strength', default=1, type=float)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--iter', default=10, type=int)
# Others
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--seed', type=int, default=222, help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
# Energy-OE
parser.add_argument('--m_in', type=float, default=-25., help='default: -25. margin for in-distribution; above this value will be penalized')
parser.add_argument('--m_out', type=float, default=-7., help='default: -7. margin for out-distribution; below this value will be penalized')
parser.add_argument('--energy_beta', default=0.1, type=float, help='beta for energy fine tuning loss')
# method
parser.add_argument('--method', type=str, default='v1', help='version')
# score_type
parser.add_argument('--score_type', type=str, default='ours', help='version')
parser.add_argument('--stage2_start', default=25, type=int)
parser.add_argument('--alpha',   default=1, type=float)
parser.add_argument('--beta',   default=1, type=float)
parser.add_argument('--model', type=str, default='resnet50', help='version')

args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args.gamma, args.beta, args.rho, args.seed)

cudnn.benchmark = True  # fire on all cylinders

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        self.log = open(logFile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


file_name=os.path.basename(__file__).split(".")[0]
os.makedirs('./'+file_name+'/', exist_ok=True)
sys.stdout = Logger(file_name+'/output.log')

###############################################################
class TinyImageNet_load(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(self.train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt

###############################################################
class CustomDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform=transform)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name, target = self.imgs[idx]
        # img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, target
        except Exception as e:
            print(f"Skipping corrupted image: {img_name}, due to {e}")
            return self.__getitem__((idx + 1) % len(self))  


class FilteredImageNet(Dataset):
    def __init__(self, root, tiny_wnids_path, transform=None):
        
        self.dataset = CustomDataset(root=root, transform=transform)
        print('imagenet-21k-p:', len(self.dataset))

        
        with open(tiny_wnids_path, 'r') as f:
            self.exclude_wnids = set(f.read().splitlines())

        self.filtered_indices = [i for i, (img, label) in enumerate(self.dataset.imgs)
                                 if self.dataset.classes[label] not in self.exclude_wnids]
    

    def __getitem__(self, index):
        return self.dataset[self.filtered_indices[index]]

    def __len__(self):
        return len(self.filtered_indices)


def subdataset(dataset_, num_):
    dataset_, _ = torch.utils.data.random_split(dataset_, [num_, len(dataset_)-num_], generator=torch.Generator().manual_seed(0))
    return dataset_

# mean and standard deviation of channels of tiny-imagenet
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224),
                        trn.ToTensor(), trn.Normalize(mean, std)])
test_transform = trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])

if args.dataset == 'cifar10':
    train_data_in = dset.CIFAR10('./data', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10('./data', train=False, transform=test_transform, download=True)
    cifar_data = dset.CIFAR100('./data', train=False, transform=test_transform, download=True)
    num_classes = 10
elif args.dataset == 'cifar100':
    train_data_in = dset.CIFAR100('./data', train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100('./data', train=False, transform=test_transform, download=True)
    cifar_data = dset.CIFAR10('./data', train=False, transform=test_transform, download=True)
    num_classes = 100
elif args.dataset == 'imagenet-1k':
    # train_data_in = torchvision.datasets.ImageFolder('/opt/data/common/ILSVRC2012/train', train_transform)  
    train_data_in = CustomDataset('./data/ILSVRC2012/train', train_transform)  
    test_data = torchvision.datasets.ImageFolder('./data/ILSVRC2012/val', test_transform)
    root_dir = './data/imagenet21k_resized/imagenet21k_train' 
    tiny_wnids_path = './data/wnids.txt'  
    ood_data = FilteredImageNet(root=root_dir, tiny_wnids_path=tiny_wnids_path, transform=train_transform)
    print('imagenet-21k-p-exclude-1k:', len(ood_data))
    print('imagenet train-data-in set:', len(train_data_in))
    num_classes = 1000


train_loader_in = torch.utils.data.DataLoader(train_data_in, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
train_loader_out = torch.utils.data.DataLoader(ood_data, batch_size=args.oe_batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

real_ood_transform = test_transform
inat_data = dset.ImageFolder(root='./data/ood_data/imagenet/iNaturalist', transform=real_ood_transform)
sun_data = dset.ImageFolder(root="./data/ood_data/imagenet/SUN", transform=real_ood_transform)
places_data = dset.ImageFolder(root="./data/ood_data/imagenet/Places", transform=real_ood_transform)
texture_data = dset.ImageFolder(root="./data/ood_data/dtd/images", transform=real_ood_transform)

inat_loader = torch.utils.data.DataLoader(inat_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
sun_loader = torch.utils.data.DataLoader(sun_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
places_loader = torch.utils.data.DataLoader(places_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)
texture_loader = torch.utils.data.DataLoader(texture_data, batch_size=args.test_bs, shuffle=True, num_workers=4, pin_memory=False)


ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))
concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()

def knn(feat_log, feat_log_val, K=5):
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    # normalizer = lambda x: (x-np.expand_dims(x.min(1), axis=1))/np.expand_dims(x.max(1)-x.min(1), axis=1)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)
    #################### KNN score OOD detection #################
    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)

    D, _ = index.search(ftest, K)
    scores_in = -D[:,-1]
    return scores_in

def get_ood_scores(loader, score_type='msp', in_dist=False, **kwargs):
    _score = []
    net.eval()
    if score_type == "maha":
        with torch.enable_grad():
            num_classes = 10 if kwargs['data']=='cifar10' else 100
            conf = mahalanobis_official(net, loader, kwargs['sample_class_mean'], kwargs['variance'], num_classes, magnitude=0.01, data=kwargs['data'], in_dist=in_dist, ood_num_examples=ood_num_examples, test_bs=args.test_bs)
            _score.append(-conf)   
    elif score_type == "knn":
        if kwargs['data'] == 'cifar10' or kwargs['data'] == 'cifar100':
            K=1
        elif kwargs['data'] == 'imagenet':
            K=10
        test_feature = []
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break
            data, target = data.cuda(), target.cuda()
            output, emb = net.pred_emb(data)
            test_feature.append(emb.cpu().detach())
        test_feature = torch.cat(test_feature, dim=0).numpy()
        conf = knn(kwargs['train_feature'], test_feature, K)
        _score.append(-conf)  
    else:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                    break
                data, target = data.cuda(), target.cuda()
                output = net(data)
                # output, emb = net.pred_emb(data)
                # print(output.shape, emb.shape)
                if score_type == 'msp':
                    smax = to_np(F.softmax(output, dim=1))
                    _score.append(-np.max(smax, axis=1))
                elif score_type == 'ours':
                    num_classes = 10 if kwargs['data']=='cifar10' else 100
                    target = torch.argmax(output.data, 1).detach()
                    emb = emb/torch.norm(emb, dim=1, keepdim=True)
                    a = net.fc.weight.data/torch.norm(net.fc.weight.data, dim=1, keepdim=True)
                    cosine = torch.norm((emb @ a.T), p=1, dim=1).cpu().detach().numpy()
                    #####
                    smax = to_np(F.softmax(output, dim=1))
                    msp = np.max(smax, axis=1)
                    #####
                    conf = msp + cosine
                    _score.append(-conf)
                elif score_type == 'energy':   
                    temper = 1
                    conf = temper * (torch.logsumexp(output / temper, dim=1))
                    _score.append(-conf.data.cpu().numpy())    

    if in_dist:
        return concat(_score).copy() # , concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def get_and_print_results(ood_loader, in_score, score_type='msp', num_to_avg=1, **kwargs):
    net.eval()
    aurocs, auprs, fprs = [], [], []
    for _ in range(num_to_avg):
        out_score = get_ood_scores(ood_loader, score_type, **kwargs)
        print('out_score.shape:', out_score.shape)
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



def finetune_with_oe(epoch, args):
    net.train()

    loss_avg = 0.0
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_in.dataset))
    for batch_idx, (in_set, out_set) in enumerate(zip(train_loader_in, train_loader_out)):
        data, target = torch.cat((in_set[0], out_set[0]), 0), in_set[1]
        data, target = data.cuda(), target.cuda()

        x = net(data)
        l_ce = F.cross_entropy(x[:len(in_set[0])], target)
        l_oe_old = - (x[len(in_set[0]):].mean(1) - torch.logsumexp(x[len(in_set[0]):], dim=1)).mean()

        loss = l_ce + 0.5*l_oe_old
        
        ############################################################################################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg = loss_avg * 0.8 + float(loss) * 0.2
        sys.stdout.write('\r epoch %2d %d/%d loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_avg))
        # sys.stdout.write('\r epoch %2d %d/%d parloss %.2f orthloss %.2f loss %.2f' %(epoch, batch_idx + 1, len(train_loader_in), loss_parallel, loss_orth, l_ce + 0.5*l_oe_old))
        scheduler.step()
    return


if args.model == 'resnet50':
    import torchvision
    from torchvision import models
    net = models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    net.cuda()
elif args.model == 'densenet':
    net = densenet121(num_classes=200).cuda()
elif args.model == 'resnet18':
    net = resnet18(200).cuda()
    

optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: cosine_annealing(step, args.epochs * len(train_loader_in), 1, 1e-6 / args.learning_rate))


# ############################ Test Mode #####################################
# net.load_state_dict(torch.load(file_name + '/models/' + args.dataset + '_' + args.model +'_finetune_with_oe.pt'))   
# net.eval()
# print('\n acc:', test())
# in_score = get_ood_scores(test_loader, in_dist=True, score_type=args.score_type, data=args.dataset)
# metric_ll = []
# metric_ll.append(get_and_print_results(inat_loader, in_score, score_type=args.score_type, data=args.dataset))
# metric_ll.append(get_and_print_results(sun_loader, in_score, score_type=args.score_type, data=args.dataset))
# metric_ll.append(get_and_print_results(places_loader, in_score, score_type=args.score_type, data=args.dataset))
# metric_ll.append(get_and_print_results(texture_loader, in_score, score_type=args.score_type, data=args.dataset))
# print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
# print(fdsfsd)
# #################################################################     

os.makedirs('./' + file_name + '/models/', exist_ok=True)
for epoch in range(args.epochs):
    finetune_with_oe(epoch, args)
    checkpoint = {
        'model':net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_name + '/models/' + str(epoch) + args.dataset + '_' + args.model +'_finetune_with_oe.pth')
   
    if epoch % 5 == 4: 
        net.eval()
        print('\n test acc:', test())
        in_score = get_ood_scores(test_loader, in_dist=True, score_type=args.score_type, data=args.dataset)
        metric_ll = []
        metric_ll.append(get_and_print_results(inat_loader, in_score, score_type=args.score_type, data=args.dataset))
        metric_ll.append(get_and_print_results(sun_loader, in_score, score_type=args.score_type, data=args.dataset))
        metric_ll.append(get_and_print_results(places_loader, in_score, score_type=args.score_type, data=args.dataset))
        metric_ll.append(get_and_print_results(texture_loader, in_score, score_type=args.score_type, data=args.dataset))
        print('\n & %.2f & %.2f & %.2f' % tuple((100 * torch.Tensor(metric_ll).mean(0)).tolist()))
    
    

