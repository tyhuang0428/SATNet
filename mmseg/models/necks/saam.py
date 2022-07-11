from math import log
import torch
import torch.nn as nn
from torchvision.transforms import RandomHorizontalFlip

from ..builder import NECKS


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, q, k):
        _, _, dim = q.shape
        scale = dim ** -0.5
        attn = (q @ k.transpose(-1, -2).contiguous()) * scale
        return attn.softmax(dim=-1)


class ConcatAttention(nn.Module):
    def __init__(self, channel):
        super(ConcatAttention, self).__init__()
        self.attn = Attention()
        self.conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x, y):
        b, c, h, w = x.shape
        cat = self.conv(torch.cat((x, y), dim=1)).reshape(b, c, -1).transpose(-1, -2).contiguous()
        x = x.reshape(b, c, -1).transpose(-1, -2).contiguous()
        y = y.reshape(b, c, -1).transpose(-1, -2).contiguous()
        attn1 = self.attn(x, cat)
        attn2 = self.attn(y, cat)
        return tuple([(attn1 @ cat).transpose(-1, -2).reshape(b, c, h, w).contiguous(),
                      (attn2 @ cat).transpose(-1, -2).reshape(b, c, h, w).contiguous(),
                      attn1, attn2])


class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, beta=1):
        super(ECABlock, self).__init__()
        t = int(abs((log(channel, 2) + beta) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2).contiguous())
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1).contiguous())
        return x * y.expand_as(x)


@NECKS.register_module()
class SAAMNeck(nn.Module):
    def __init__(self, reverse=True, channel=(768, 384)):
        super(SAAMNeck, self).__init__()
        self.reverse = reverse
        self.attn1 = ConcatAttention(channel[0])
        self.attn2 = ConcatAttention(channel[1])

        if self.reverse:
            self.reverse_attn1 = ConcatAttention(channel[0])
            self.reverse_attn2 = ConcatAttention(channel[1])

        self.flip = RandomHorizontalFlip(p=1)

        self.eca1 = ECABlock(channel[0])
        self.eca2 = ECABlock(channel[1])
        self.reverse_eca1 = ECABlock(channel[0])
        self.reverse_eca2 = ECABlock(channel[1])

    def forward(self, inputs, reverses):
        front1 = inputs[-1]
        back1 = self.flip(reverses[-1])
        feats1 = self.attn1(front1, back1)
        front_attn1 = self.eca1(feats1[0])

        front2 = inputs[-2]
        back2 = self.flip(reverses[-2])
        feats2 = self.attn2(front2, back2)
        front_attn2 = self.eca2(feats2[0])

        results = []
        for i in range(len(inputs) - 2):
            results.append(inputs[i])
        results.append(front_attn2)
        results.append(front_attn1)
        if self.reverse:
            back_attn1 = self.reverse_eca1(feats1[1])
            back_attn2 = self.reverse_eca2(feats2[1])
            for i in range(len(inputs) - 2):
                results.append(self.flip(reverses[i]))
            results.append(back_attn2)
            results.append(back_attn1)

        return tuple(results)
