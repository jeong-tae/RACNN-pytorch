import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .vgg import vgg19_bn

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
            nn.Linear(512 * 14 * 14, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3),
            nn.Sigmoid(),
        )
        self.crop_resize = AttentionCropLayer()

        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(512, num_classes)

    def forward(self, x):

        conv5_4 = self.b1.features[:-1](x)
        pool5 = self.feature_pool1(conv5_4)
        atten1 = self.apn1(self.atten_pool(conv5_4).view(-1, 512 * 14 * 14))
        scaledA_x = self.crop_resize(x, atten1 * 448)
        # scaledA_x: (224, 224) size image

        conv5_4_A = self.b2.features[:-1](scaledA_x)
        pool5_A = self.feature_pool2(conv5_4_A)
        atten2 = self.apn2(conv5_4_A.view(-1, 512 * 14 * 14))
        scaledAA_x = self.crop_resize(scaledA_x, atten2 * 224)

        pool5_AA = self.feature_pool2(self.b3.features[:-1](scaledAA_x))

        pool5 = pool5.view(-1, 512)
        pool5_A = pool5_A.view(-1, 512)
        pool5_AA = pool5_AA.view(-1, 512)

        """#Feature fusion
        scale123 = torch.cat([pool5, pool5_A, pool5_AA], 1)
        scale12 = torch.cat([pool5, pool5_A], 1)
        """

        logits1 = self.classifier1(pool5)
        logits2 = self.classifier2(pool5_A)
        logits3 = self.classifier3(pool5_AA)
        return [logits1, logits2, logits3], [conv5_4, conv5_4_A], [atten1, atten2], [scaledA_x, scaledAA_x]

class AttentionCropFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, images, locs):
        h = lambda x: 1. / (1. + torch.exp(-10. * x))
        in_size = images.size()[2]
        unit = torch.stack([torch.arange(0, in_size)] * in_size).float()
        x = torch.stack([unit.t()] * 3)
        y = torch.stack([unit] * 3)
        if isinstance(images, torch.cuda.FloatTensor):
            x, y = x.cuda(), y.cuda()
        
        in_size = images.size()[2]
        ret = []
        for i in range(images.size(0)):
            tx, ty, tl = locs[i][0], locs[i][1], locs[i][2]
            tl = tl if tl > (in_size/3) else in_size/3
            tx = tx if tx > tl else tl
            tx = tx if tx < in_size-tl else in_size-tl
            ty = ty if ty > tl else tl
            ty = ty if ty < in_size-tl else in_size-tl

            w_off = int(tx-tl) if (tx-tl) > 0 else 0
            h_off = int(ty-tl) if (ty-tl) > 0 else 0
            w_end = int(tx+tl) if (tx+tl) < in_size else in_size
            h_end = int(ty+tl) if (ty+tl) < in_size else in_size

            mk = (h(x-w_off) - h(x-w_end)) * (h(y-h_off) - h(y-h_end))
            xatt = images[i] * mk
            
            xatt_cropped = xatt[:, w_off:w_end, h_off:h_end]
            before_upsample = Variable(xatt_cropped.unsqueeze(0))
            xamp = F.interpolate(before_upsample, size=(224,224), mode='bilinear', align_corners = True)
            ret.append(xamp.data.squeeze())
        
        ret_tensor = torch.stack(ret)
        self.save_for_backward(images, ret_tensor)
        return ret_tensor

    @staticmethod
    def backward(self, grad_output):
        images, ret_tensor = self.saved_variables[0], self.saved_variables[1]
        in_size = 224
        ret = torch.Tensor(grad_output.size(0), 3).zero_()
        norm = -(grad_output * grad_output).sum(dim=1)
        
#         show_image(inputs.cpu().data[0])
#         show_image(ret_tensor.cpu().data[0])
#         plt.imshow(norm[0].cpu().numpy(), cmap='gray')
        
        x = torch.stack([torch.arange(0, in_size)] * in_size).t()
        y = x.t()
        long_size = (in_size/3*2)
        short_size = (in_size/3)
        mx = (x >= long_size).float() - (x < short_size).float()
        my = (y >= long_size).float() - (y < short_size).float()
        ml = (((x<short_size)+(x>=long_size)+(y<short_size)+(y>=long_size)) > 0).float()*2 - 1
        
        mx_batch = torch.stack([mx.float()] * grad_output.size(0))
        my_batch = torch.stack([my.float()] * grad_output.size(0))
        ml_batch = torch.stack([ml.float()] * grad_output.size(0))
        
        if isinstance(grad_output, torch.cuda.FloatTensor):
            mx_batch = mx_batch.cuda()
            my_batch = my_batch.cuda()
            ml_batch = ml_batch.cuda()
            ret = ret.cuda()
        
        ret[:, 0] = (norm * mx_batch).sum(dim=1).sum(dim=1)
        ret[:, 1] = (norm * my_batch).sum(dim=1).sum(dim=1)
        ret[:, 2] = (norm * ml_batch).sum(dim=1).sum(dim=1)
        return None, ret

class AttentionCropLayer(nn.Module):
    """
        Crop function sholud be implemented with the nn.Function.
        Detailed description is in 'Attention localization and amplification' part.
        Forward function will not changed. backward function will not opearate with autograd, but munually implemented function
    """
    def forward(self, images, locs):
        return AttentionCropFunction.apply(images, locs)

if __name__ == '__main__':
    print(" [*] RACNN forward test...")
    x = torch.randn([2, 3, 448, 448])
    net = RACNN(num_classes = 200)
    logits, conv5s, attens = net(x)
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
    loss.backward()
    print(" [*] Backward done")
