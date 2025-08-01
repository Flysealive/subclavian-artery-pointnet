#!/usr/bin/env python3

import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from subclavian_dataset import SubclavianDataset
import torch.nn.functional as F
import sys
sys.path.append('./pointnet.pytorch')
from pointnet.model import PointNetCls, feature_transform_regularizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='number of points')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outdir', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset_path', type=str, default='./', help='dataset path')
parser.add_argument('--numpy_dir', type=str, default='numpy_arrays', help='numpy arrays directory')
parser.add_argument('--csv_file', type=str, default='classification_labels.csv', help='labels CSV file')
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Create datasets
dataset = SubclavianDataset(
    numpy_dir=opt.numpy_dir,
    csv_file=opt.csv_file,
    split='train',
    npoints=opt.num_points
)

test_dataset = SubclavianDataset(
    numpy_dir=opt.numpy_dir,
    csv_file=opt.csv_file,
    split='val',
    npoints=opt.num_points,
    data_augmentation=False
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
)

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers)
)

print(f"Training samples: {len(dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: 2")

try:
    os.makedirs(opt.outdir)
except OSError:
    pass

# Binary classification (2 classes)
classifier = PointNetCls(k=2, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Use CPU instead of CUDA
device = torch.device('cpu')
classifier = classifier.to(device)

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        target = target[:, 0]  # Remove extra dimension from target
        points = points.transpose(2, 1)
        points, target = points.to(device), target.to(device)
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]  # Remove extra dimension from target
            points = points.transpose(2, 1)
            points, target = points.to(device), target.to(device)
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outdir, epoch))

# Final evaluation on full test set
total_correct = 0
total_testset = 0
class_correct = [0, 0]
class_total = [0, 0]

for i, data in enumerate(testdataloader, 0):
    points, target = data
    target = target[:, 0]  # Remove extra dimension from target
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
    
    # Per-class accuracy
    for j in range(points.size()[0]):
        label = target[j]
        class_total[label] += 1
        if pred_choice[j] == label:
            class_correct[label] += 1

print("Final results:")
print(f"Overall Accuracy: {total_correct / float(total_testset):.4f}")
for i in range(2):
    if class_total[i] > 0:
        print(f"Class {i} Accuracy: {class_correct[i] / float(class_total[i]):.4f} ({class_correct[i]}/{class_total[i]})")
    else:
        print(f"Class {i}: No samples")