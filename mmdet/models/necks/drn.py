import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

from ..builder import NECKS

@NECKS.register_module()
class DilaRMoudle(nn.Module):
    def __init__(self,
                 in_channel,
                 num_levels,
                 conv_cfg=None,
                 norm_cfg=None,
                 dcn_cfg=None,
                 share=False):
        super(DilaRMoudle, self).__init__()
        assert in_channel % 2**2 == 0

        self.in_channel = in_channel
        self.num_levels = num_levels
        self.conv_cfg   = conv_cfg
        self.norm_cfg   = norm_cfg
        assert dcn_cfg is None or isinstance(dcn_cfg, dict)
        self.dcn_cfg    = dcn_cfg
        self.with_dcn   = True if dcn_cfg is not None else False
        self.share      = share

        self.level_convs = nn.ModuleList()

        loop = 1 if self.share else self.num_levels

        for i in range(loop):
            convs = nn.ModuleList()

            act_cfg = dict(type='ReLU')
            conv = ConvModule(self.in_channel, self.in_channel, 3, padding=1, dilation=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg, act_cfg=act_cfg)
            convs.append(conv)

            act_cfg = dict(type='ReLU')
            conv = ConvModule(self.in_channel, self.in_channel, 3, padding=3, dilation=3, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg, act_cfg=act_cfg)
            convs.append(conv)

            act_cfg = dict(type='ReLU')
            conv = ConvModule(self.in_channel, self.in_channel, 3, padding=5, dilation=5, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg, act_cfg=act_cfg)
            convs.append(conv)

            act_cfg = dict(type='ReLU')
            conv = ConvModule(self.in_channel*4, self.in_channel, 1, padding=0, dilation=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg, act_cfg=act_cfg)
            convs.append(conv)
            
            self.level_convs.append(convs)

        if self.with_dcn:
            self.dcn = ConvModule(
                self.in_channel,
                self.in_channel,
                3,
                padding=1,
                conv_cfg=self.dcn_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=act_cfg)

    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, input):
        outs = []
        for i in range(self.num_levels):
            x = input[i]
            if self.share:
                x_1 = self.level_convs[0][0](x)
                x_2 = self.level_convs[0][1](x_1)
                x_3 = self.level_convs[0][2](x_2)
                out = torch.cat([x, x_1, x_2, x_3], dim=1)
                out = self.level_convs[0][3](out)
            else:
                x_1 = self.level_convs[i][0](x)
                x_2 = self.level_convs[i][1](x_1)
                x_3 = self.level_convs[i][2](x_2)
                out = torch.cat([x, x_1, x_2, x_3], dim=1)
                out = self.level_convs[i][3](out)
                
            if self.with_dcn:
                out = self.dcn(out)

            outs.append(out)

        return tuple(outs)
