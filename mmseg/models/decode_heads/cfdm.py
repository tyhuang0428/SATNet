import torch
import torch.nn as nn

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class CCL(nn.Module):
    def __init__(self, channel, dilation):
        super(CCL, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return self.relu(self.bn(x1 - x2))


class HighLayer(nn.Module):
    def __init__(self, channel, dilation=2):
        super(HighLayer, self).__init__()
        self.ccl_x = CCL(channel, dilation)
        self.ccl_y = CCL(channel, dilation)
        self.ccl_c = CCL(channel, dilation)
        self.concat1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.concat2 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x, y):
        cat = self.ccl_c(self.concat1(torch.cat((x, y), dim=1)))
        x_ccl = self.ccl_x(x)
        y_ccl = self.ccl_y(y)
        return self.concat2(torch.cat((x_ccl, y_ccl, cat), dim=1))


class LowLayer(nn.Module):
    def __init__(self, channel, high_channel, dilation):
        super(LowLayer, self).__init__()
        self.ccl_x = CCL(channel, dilation)
        self.ccl_y = CCL(channel, dilation)
        self.ccl_c = CCL(channel, dilation)
        self.conv_h = nn.Sequential(
            nn.Conv2d(high_channel, channel, 1),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.concat1 = nn.Sequential(
            nn.Conv2d(channel * 2, channel, 1),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.concat2 = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 1),
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, x, y, hl):
        cat = self.concat1(torch.cat((x, y), dim=1))
        hl = resize(
            hl, size=x.shape[2:], mode='bilinear', align_corners=False
        )
        hl = self.conv_h(hl)
        x_ccl = self.ccl_x(x + hl)
        y_ccl = self.ccl_y(y + hl)
        cat_ccl = self.ccl_c(cat + hl)
        return self.concat2(torch.cat((x_ccl, y_ccl, cat_ccl), dim=1))


@HEADS.register_module()
class CFDMHead(BaseDecodeHead):
    def __init__(self, **kwargs):
        super(CFDMHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.high = HighLayer(self.in_channels[-1])
        self.lows = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.lows.append(LowLayer(self.in_channels[i], self.in_channels[i + 1], 2 * (i + 2)))
            self.cls_convs.append(nn.Conv2d(self.in_channels[i], self.num_classes, 1))
        self.cls_convs.append(nn.Conv2d(self.in_channels[-1], self.num_classes, 1))

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = []
        for seg_logit in seg_logits:
            losses.append(self.losses(seg_logit, gt_semantic_seg))
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)[-1]

    def forward(self, inputs):
        x = self._transform_inputs(inputs[0])
        y = self._transform_inputs(inputs[1])
        results = []
        hl = self.high(x[-1], y[-1])
        results.append(self.cls_convs[-1](hl))
        for i in range(2, -1, -1):
            hl = self.lows[i](x[i], y[i], hl)
            results.append(self.cls_convs[i](hl))
        return tuple(results)