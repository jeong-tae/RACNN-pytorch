import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vgg import vgg19_bn

import time
class RACNN(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(RACNN, self).__init__()
        if pretrained:
            pretrained = 'imagenet'
        else:
            pretrained = False # make sure

        self.b1 = vgg19_bn(num_classes = 1000, pretrained = 'imagenet')
        self.b2 = vgg19_bn(num_classes = 1000, pretrained = 'imagenet')
        self.b3 = vgg19_bn(num_classes = 1000, pretrained = 'imagenet')

        self.feature_pool1 = nn.AvgPool2d(kernel_size = 28, stride = 28)
        self.feature_pool2 = nn.AvgPool2d(kernel_size = 14, stride = 14)
        
        self.atten_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.apn1 = nn.Sequential(
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )

        self.apn2 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.crop_resize = CropAndResize(out_size = 224)

        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512 * 2, num_classes)
        self.classifier3 = nn.Linear(512 * 3, num_classes)

    def forward(self, x):

        conv5_4 = self.b1.features[:-1](x)
        pool5 = self.feature_pool1(conv5_4)
        atten1 = self.apn1(self.atten_pool(conv5_4).view(-1, 512 * 14 * 14))
        scaledA_x = self.crop_resize(x, atten1 * 448)

        conv5_4_A = self.b2.features[:-1](scaledA_x)
        pool5_A = self.feature_pool2(conv5_4_A)
        atten2 = self.apn2(conv5_4_A.view(-1, 512 * 7 * 7))
        scaledAA_x = self.crop_resize(scaledA_x, atten2 * 224)

        pool5_AA = self.feature_pool2(self.b3.features[:-1](scaledAA_x))

        pool5 = pool5.view(-1, 512) * 0.1
        pool5_A = pool5_A.view(-1, 512) * 0.1
        pool5_AA = pool5_AA.view(-1, 512) * 0.1

        scale123 = torch.cat([pool5, pool5_A, pool5_AA], 1)
        scale12 = torch.cat([pool5, pool5_A], 1)

        logits1 = self.classifier1(pool5)
        logits2 = self.classifier2(scale12)
        logits3 = self.classifier3(scale123)
        return [logits1, logits2, logits3]

class CropLayer(torch.autograd.Function):
    @staticmethod
    def forward(self, image, loc):
        self.save_for_backward(image, loc)
        #self.save_for_backward(loc)
        in_size = image.size()[1]
        tx, ty, tl = loc[0], loc[1], loc[2]
        tl = tl if tl > 0.01 * in_size else 0.01 * in_size

        w_off = int(tx-tl) if (tx-tl) > 0 else 0
        h_off = int(ty-tl) if (ty-tl) > 0 else 0
        w_end = int(tx+tl) if (tx+tl) < in_size else in_size
        h_end = int(ty+tl) if (ty+tl) < in_size else in_size
        cropped = image[:, h_off:h_end, w_off:w_end]
        print(" [*] h_off, h_end:", h_off, h_end)
        print(" [*] w_off, w_end:", w_off, w_end)
        return cropped

    @staticmethod
    def backward(self, grad_output):
        image, loc = self.saved_tensors
        grad_input = grad_output.clone()
        in_size = image.size()[1]
        tx, ty, tl = loc[0], loc[1], loc[2]

        H = lambda x: 1/(1 + torch.exp(-0.05 * x))
        diff_H = lambda x: 0.05 * torch.exp(-0.05 * x) / ((1 + torch.exp(-0.05 * x)) * (1 + torch.exp(-0.05 * x)))
        F = lambda a, b, c, x, y:(H(x - (a - c)) - H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))
        diff_F_a = lambda a, b, c, x, y: (diff_H(x - (a - c)) - diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))
        diff_F_b = lambda a, b, c, x, y: (diff_H(y - (b - c)) - diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c)))
        diff_F_c = lambda a, b, c, x, y: -((diff_H(y - (b - c)) + diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c))) + (diff_H(x - (a - c)) + diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))) + 0.005
        
        diff_loc = torch.zeros(loc.size())
        diff_output = torch.zeros(image.size())
        max_diff = torch.abs(grad_input).max()

        w_off = int(tx-tl) if (tx-tl) > 0 else 0
        h_off = int(ty-tl) if (ty-tl) > 0 else 0
        w_end = int(tx+tl) if (tx+tl) < in_size else in_size
        h_end = int(ty+tl) if (ty+tl) < in_size else in_size
        diff_output[:, h_off:h_end, w_off:w_end] = grad_input

        tops = diff_output
        if max_diff > 0: # divide first can reduce the duplicated computation
            tops = tops / max_diff * 0.0000001
        
        tops = tops.sum(0) # channel sum. after sum also can get same result
        size_range = torch.range(0, tops.size()[1]-1)
        xs = tx - tl + 2 * size_range * tl / 224 # 224 is fixed size, don't know why...
        ys = ty - tl + 2 * size_range * tl / 224
        t0 = time.time()

        xs = xs.expand(tops.size()[0], tops.size()[1])
        ys = ys.expand(tops.size()[0], tops.size()[1])
        diff_loc[0] = torch.sum(tops * diff_F_a(tx, ty, tl, xs, ys))
        diff_loc[1] = torch.sum(tops * diff_F_b(tx, ty, tl, xs, ys))
        diff_loc[2] = torch.sum(tops * diff_F_c(tx, ty, tl, xs, ys))
        t1 = time.time()
        print(" [*] loc grad time: %.4fsec"%(t1-t0))

        return diff_output, diff_loc


class CropAndResize(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def __init__(self, out_size):
        super(CropAndResize, self).__init__()
        self.out_size = out_size
        self.crop = CropLayer.apply

    def forward(self, images, locs):
        N = images.size()[0]
        
        outputs = []
        for i in range(N):
            cropped = self.crop(images[i], locs[i])
            resized = F.upsample(cropped.unsqueeze(0), size = [self.out_size, self.out_size], mode = 'bilinear', align_corners = True)
            outputs.append(resized)

        outputs = torch.cat(outputs, 0)
        return outputs

if __name__ == '__main__':
    print(" [*] RACNN forward test...")
    x = torch.randn([2, 3, 448, 448])
    net = RACNN(num_classes = 200)
    logits = net(x)
    print(" [*] logits[0]:", logits[0].size())

    from Loss import multitask_loss, pairwise_ranking_loss
    target_cls = torch.LongTensor([100, 150])

    preds = []
    for i in range(len(target_cls)):
        pred = [logit[i][target_cls[i]] for logit in logits]
        preds.append(pred)
    loss_cls = multitask_loss(logits, target_cls)
    loss_rank = pairwise_ranking_loss(preds)
    print(" [*] Loss cls:", loss_cls)
    print(" [*] Loss rank:", loss_rank)

    print(" [*] Backward test")
    loss = loss_cls + loss_rank
    t0 = time.time()
    loss.backward()
    t1 = time.time()
    print(" [*] Backward time: %.4f"%(t1-t0))
    print(" [*] Backward done")
