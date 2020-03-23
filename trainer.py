from visual import Logger

import os
import os.path as osp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F

from data import CUB200_loader
import cv2

from models import RACNN, pairwise_ranking_loss, multitask_loss
from utils import save_img

import argparse

parser = argparse.ArgumentParser(description = 'Training arguments')
parser.add_argument('--cuda', default = True, type = bool, help = "use cuda to train")
parser.add_argument('--lr', default = 0.01, type = float, help = "initial learning rate")
args = parser.parse_args()
decay_steps = [20, 40] # based on epoch

net = RACNN(num_classes = 200)
if args.cuda and torch.cuda.is_available():
    print(" [*] Set cuda: True")
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
else:
    print(" [*] Set cuda: False")

logger = Logger('./visual/' + 'RACNN_CUB200_9')
cls_params = list(net.module.b1.parameters()) + list(net.module.b2.parameters()) + list(net.module.b3.parameters()) + list(net.module.classifier1.parameters()) + list(net.module.classifier2.parameters()) + list(net.module.classifier3.parameters())

opt1 = optim.SGD(cls_params, lr = args.lr, momentum = 0.9, weight_decay = 0.0005)

apn_params = list(net.module.apn1.parameters()) + list(net.module.apn2.parameters())
opt2 = optim.SGD(apn_params, lr = args.lr, momentum = 0.9, weight_decay = 0.0005)
#for param in apn_params:
#    param.register_hook(print)

def train():
    net.train()

    conf_loss = 0
    loc_loss = 0

    print(" [*] Loading dataset...")
    batch_iterator = None
    
    trainset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split = 'train')
    trainloader = data.DataLoader(trainset, batch_size = 4,
            shuffle = True, collate_fn = trainset.CUB_collate, num_workers = 4)
    testset = CUB200_loader(os.getcwd() + '/data/CUB_200_2011', split = 'test')
    testloader = data.DataLoader(testset, batch_size = 4,
            shuffle = False, collate_fn = testset.CUB_collate, num_workers = 4)
    test_sample, _ = next(iter(testloader))

    apn_iter, apn_epoch, apn_steps = pretrainAPN(trainset, trainloader)
    cls_iter, cls_epoch, cls_steps = 0, 0, 1
    switch_step = 0
    old_cls_loss, new_cls_loss = 2, 1
    old_apn_loss, new_apn_loss = 2, 1
    iteration = 0 # count the both of iteration
    epoch_size = len(trainset) // 4
    cls_tol = 0
    apn_tol = 0
    batch_iterator = iter(trainloader)

    while ((old_cls_loss - new_cls_loss)**2 > 1e-7) and ((old_apn_loss - new_apn_loss)**2 > 1e-7) and (iteration < 500000):
        # until the two type of losses no longer change
        print(' [*] Swtich optimize parameters to Class')
        while ((cls_tol < 10) and (cls_iter % 5000 != 0)):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if cls_iter % epoch_size == 0:
                cls_epoch += 1
                if cls_epoch in decay_steps:
                    cls_steps += 1
                    adjust_learning_rate(opt1, 0.1, cls_steps, args.lr)

            old_cls_loss = new_cls_loss

            images, labels = next(batch_iterator)
            images, labels = Variable(images, requires_grad = True), Variable(labels)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            t0 = time.time()
            logits, _, _, _ = net(images)
            
            opt1.zero_grad()
            new_cls_losses = multitask_loss(logits, labels)
            new_cls_loss = sum(new_cls_losses)
            #new_cls_loss = new_cls_losses[0]
            new_cls_loss.backward()
            opt1.step()
            t1 = time.time()

            if (old_cls_loss - new_cls_loss)**2 < 1e-6:
                cls_tol += 1
            else:
                cls_tol = 0


            logger.scalar_summary('cls_loss', new_cls_loss.item(), iteration + 1)
            logger.scalar_summary('cls_loss1', new_cls_losses[0].item(), iteration + 1)
            logger.scalar_summary('cls_loss12', new_cls_losses[1].item(), iteration + 1)
            logger.scalar_summary('cls_loss123', new_cls_losses[2].item(), iteration + 1)
            iteration += 1
            cls_iter += 1
            if (cls_iter % 20) == 0:
                print(" [*] cls_epoch[%d], Iter %d || cls_iter %d || cls_loss: %.4f || Timer: %.4fsec"%(cls_epoch, iteration, cls_iter, new_cls_loss.item(), (t1 - t0)))

        images, labels = next(batch_iterator)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        logits, _, _ = net(images)
        preds = []
        for i in range(len(labels)):
            pred = [logit[i][labels[i]] for logit in logits]
            preds.append(pred)
        new_apn_loss = pairwise_ranking_loss(preds)
        logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
        iteration += 1
        #cls_iter += 1
        test(testloader, iteration)
        #continue
        print(' [*] Swtich optimize parameters to APN')
        switch_step += 1

        while ((apn_tol < 10) and apn_iter % 5000 != 0):
            if (not batch_iterator) or (iteration % epoch_size == 0):
                batch_iterator = iter(trainloader)

            if apn_iter % epoch_size == 0:
                apn_epoch += 1
                if apn_epoch in decay_steps:
                    apn_steps += 1
                    adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)

            old_apn_loss = new_apn_loss

            images, labels = next(batch_iterator)
            images, labels = Variable(images, requires_grad = True), Variable(labels)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            t0 = time.time()
            logits, _, _, _ = net(images)

            opt2.zero_grad()
            preds = []
            for i in range(len(labels)):
                pred = [logit[i][labels[i]] for logit in logits]
                preds.append(pred)
            new_apn_loss = pairwise_ranking_loss(preds)
            new_apn_loss.backward()
            opt2.step()
            t1 = time.time()

            if (old_apn_loss - new_apn_loss)**2 < 1e-6:
                apn_tol += 1
            else:
                apn_tol = 0

            logger.scalar_summary('rank_loss', new_apn_loss.item(), iteration + 1)
            iteration += 1
            apn_iter += 1
            if (apn_iter % 20) == 0:
                print(" [*] apn_epoch[%d], Iter %d || apn_iter %d || apn_loss: %.4f || Timer: %.4fsec"%(apn_epoch, iteration, apn_iter, new_apn_loss.item(), (t1 - t0)))

        switch_step += 1

        images, labels = next(batch_iterator)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        logits, _, _ = net(images)
        new_cls_losses = multitask_loss(logits, labels)
        new_cls_loss = sum(new_cls_losses)
        logger.scalar_summary('cls_loss', new_cls_loss.item(), iteration + 1)
        iteration += 1
        cls_iter += 1
        apn_iter += 1
        test(testloader, iteration)
        _, _, _, crops = net(test_sample)
        x1, x2 = crops[0].data, crops[1].data
        # visualize cropped inputs
        save_img(x1, path=f'samples/iter_{iteration}@2x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {iteration}')
        save_img(x2, path=f'samples/iter_{iteration}@4x.jpg', annotation=f'loss = {avg_loss:.7f}, step = {iteration}')

def pretrainAPN(trainset, trainloader):
    epoch_size = len(trainset) // 4
    apn_steps, apn_epoch = 1, -1

    batch_iterator = iter(trainloader)
    for _iter in range(0, 20000):
        iteration = _iter
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(trainloader)
        
        if iteration % epoch_size == 0:
            apn_epoch += 1
            if apn_epoch in decay_steps:
                apn_steps += 1
                adjust_learning_rate(opt2, 0.1, apn_steps, args.lr)

        images, labels = next(batch_iterator)
        images, labels = Variable(images, requires_grad = True), Variable(labels)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()

        t0 = time.time()
        _, conv5s, attens, _ = net(images)
            
        opt2.zero_grad()
        # search regions with the highest response value in conv5
        weak_loc = []
        for i in range(len(conv5s)):
            loc_label = torch.ones([images.size(0),3]) * 0.33 # tl = 0.25, fixed
            resize = 448
            if i >= 1:
                resize = 224
            if args.cuda:
                loc_label = loc_label.cuda()
            for j in range(images.size(0)):
                response_map = conv5s[i][j]
                response_map = F.upsample(response_map, size = [resize, resize])
                response_map = response_map.mean(0)
                rawmaxidx = response_map.view(-1).max(0)[1]
                idx = []
                for d in list(response_map.size())[::-1]:
                    idx.append(rawmaxidx % d)
                    rawmaxidx = rawmaxidx / d
                loc_label[j, 0] = (idx[1].float() + 0.5) / response_map.size(0)
                loc_label[j, 1] = (idx[0].float() + 0.5) / response_map.size(1)
            weak_loc.append(loc_label)
        weak_loss1 = F.smooth_l1_loss(attens[0], weak_loc[0])
        weak_loss2 = F.smooth_l1_loss(attens[1], weak_loc[1])
        apn_loss = weak_loss1 + weak_loss2
        apn_loss.backward()
        opt2.step()
        t1 = time.time()
        
        if (iteration % 20) == 0:
            print(" [*] pre_apn_epoch[%d], || pre_apn_iter %d || pre_apn_loss: %.4f || Timer: %.4fsec"%(apn_epoch, iteration, apn_loss.item(), (t1 - t0)))

        logger.scalar_summary('pre_apn_loss', apn_loss.item(), iteration + 1)

    return 20000, apn_epoch, apn_steps

def test(testloader, iteration):
    net.eval()
    with torch.no_grad():
        corrects1 = 0
        corrects2 = 0
        corrects3 = 0
        cnt = 0
        test_cls_losses = []
        test_apn_losses = []
        for test_images, test_labels in testloader:
            if args.cuda:
                test_images = test_images.cuda()
                test_labels = test_labels.cuda()
            cnt += test_labels.size(0)

            logits, _, _, _ = net(test_images)

            preds = []
            for i in range(len(test_labels)):
                pred = [logit[i][test_labels[i]] for logit in logits]
                preds.append(pred)
            test_cls_losses = multitask_loss(logits, test_labels)
            test_apn_loss = pairwise_ranking_loss(preds)
            test_cls_losses.append(sum(test_cls_losses))
            test_apn_losses.append(test_apn_loss)
            _, predicted1 = torch.max(logits[0], 1)
            correct1 = (predicted1 == test_labels).sum()
            corrects1 += correct1
            _, predicted2 = torch.max(logits[1], 1)
            correct2 = (predicted2 == test_labels).sum()
            corrects2 += correct2
            _, predicted3 = torch.max(logits[2], 1)
            correct3 = (predicted3 == test_labels).sum()
            corrects3 += correct3

        test_cls_losses = torch.stack(test_cls_losses).mean()
        test_apn_losses = torch.stack(test_apn_losses).mean()
        accuracy1 = corrects1.float() / cnt
        accuracy2 = corrects2.float() / cnt
        accuracy3 = corrects3.float() / cnt
        logger.scalar_summary('test_cls_loss', test_cls_losses.item(), iteration + 1)
        logger.scalar_summary('test_rank_loss', test_apn_losses.item(), iteration + 1)
        logger.scalar_summary('test_acc1', accuracy1.item(), iteration + 1)
        logger.scalar_summary('test_acc2', accuracy2.item(), iteration + 1)
        logger.scalar_summary('test_acc3', accuracy3.item(), iteration + 1)
        print(" [*] Iter %d || Test accuracy1: %.4f, Test accuracy2: %.4f, Test accuracy3: %.4f"%(iteration, accuracy1.item(), accuracy2.item(), accuracy3.item()))

    net.train()

def adjust_learning_rate(optimizer, gamma, steps, _lr):
    lr = _lr * (gamma ** (steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()
    print(" [*] Train done")            

