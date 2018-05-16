import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vgg import vgg19_bn


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

        self.apn1 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
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

        self.classifier1 = nn.Linear(512, 200)
        self.classifier2 = nn.Linear(512 * 2, 200)
        self.classifier3 = nn.Linear(512 * 3, 200)


    def forward(self, x):

        conv5_4 = self.b1.features[:-1](x)
        pool5 = self.feature_pool1(conv5_4)
        atten1 = self.apn1(conv5_4)
        scaledA_x = self.crop_resize(x, atten1 * 448)

        conv5_4_A = self.b2.features[:-1](scaledA_x)
        pool5_A = self.feature_pool2(conv5_4_A)
        atten2 = self.apn2(conv5_4_A)
        scaledAA_x = self.crop_resize(scaledA_X, atten2 * 224)

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
        cropped = image[:, w_off:w_end, h_off:h_end]
        cropped = cropped.view(1, 3, w_end-w_off, h_end-h_off)
        return cropped

    def backward(self, grad_output):
        loc = self.saved_tensors
        grad_input = grad_output.clone()
        tx, ty, tl = loc[0], loc[1], loc[2]
        H = lambda x: 1/(1 + torch.exp(-0.05 * x))
        diff_H = lambda x: 0.05 * torch.exp(-0.05 * x) / ((1 + torch.exp(-0.05 * x)) * (1 + torch.exp(-0.05 * x)))

        F = lambda a, b, c, x, y:(H(x - (a - c)) - H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))
        diff_F_a = lambda a, b, c, x, y: (diff_H(x - (a - c)) - diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))
        diff_F_b = lambda a, b, c, x, y: (diff_H(y - (b - c)) - diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c)))
        diff_F_c = lambda a, b, c, x, y: -((diff_H(y - (b - c)) + diff_H(y - (b + c)))*(H(x - (a - c)) - H(x - (a + c))) + (diff_H(x - (a - c)) + diff_H(x - (a + c)))*(H(y - (b - c)) - H(y - (b + c)))) + 0.005

        k, l = grad_input.size()

        import pdb
        pdb.set_trace()

        return None, loc


class CropAndResize(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def __init__(self, out_size):
        super(CropAndResize, self).__init__()
        self.out_size = out_size

    def forward(self, images, locs):
        N = images.size()[0]
        
        outputs = []
        for i in range(N):
            cropped = CropLayer(images[i], locs[i])
            resized = F.upsample(cropped, size = [self.out_size, self.out_size], mode = 'nilinear')
            outputs.append(resized)

        outputs = torch.stack(outputs, 0)
        return outputs

