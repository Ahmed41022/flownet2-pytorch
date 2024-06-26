import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np
import torch.nn.functional as F


from networks import FlowNetC
from networks import FlowNetS
from networks import FlowNetSD
from networks import FlowNetFusion
import torchvision.transforms as transforms
from networks.submodules import *
'Parameter count = 162,518,834'


# class FlowNet2(nn.Module):


class FlowNet2(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.):
        super(FlowNet2, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = nn.BatchNorm2d(1)
        # self.channelnorm = channelnorm(input, 2, dim=1, keepdim=True)
        # Block (FlowNetC)
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.resample1 = nn.functional.interpolate
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        # self.resample2 = nn.functional.interpolate

        # Block (FlowNetS2)
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)

        # Block (FlowNetSD)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')

        # self.resample3 = nn.functional.interpolate
        # self.resample4 = nn.functional.interpolate

        # Block (FLowNetFusion)
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args,
                                                         batchNorm=self.batchNorm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform(m.bias)
                init.xavier_uniform(m.weight)
                # init_deconv_bilinear(m.weight)

    def channelnorm(self, input):
        return torch.linalg.norm(input, 2, dim=1, keepdim=True)

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.)
        for i in range(min_dim):
            weight.data[i, i, :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(
            inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1, 1, 1,))

        rgb_mean = inputs.mean(dim=1, keepdim=True)
        x = (inputs - rgb_mean) / self.rgb_max
        # 255. = rgb_max
        x1 = x[:, :, 0, :, :]
        x2 = x[:, :, 1, :, :]

        x = torch.cat((x1, x2), dim=1)

        # flownetc
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2*self.div_flow)
        # warp img1 to img0; magnitude of diff between img0 and and warped_img1,
        # resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        resampled_img1 = F.interpolate(
            x[:, 3:, :, :], size=flownetc_flow.size()[-2:], mode='nearest')
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        # concat img0, img1, img1->img0, flow, diff-mag ;
        concat1 = torch.cat(
            (x, resampled_img1, flownetc_flow/self.div_flow, norm_diff_img0), dim=1)
        # flownets1
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2*self.div_flow)

        # warp img1 to img0 using flownets1; magnitude of diff between img0 and and warped_img1
        resampled_img1 = F.interpolate(
            x[:, 3:, :, :], size=flownets1_flow.size()[-2:], mode='nearest')
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-mag
        concat2 = torch.cat(
            (x, resampled_img1, flownets1_flow/self.div_flow, norm_diff_img0), dim=1)

        # flownets2
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)

        diff_flownets2_flow = F.interpolate(
            x[:, 3:, :, :], size=flownets2_flow.size()[-2:], mode='nearest')

        diff_flownets2_img1 = self.channelnorm(
            (x[:, :3, :, :]-diff_flownets2_flow))

        # flownetsd
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)

        diff_flownetsd_flow = F.interpolate(
            x[:, 3:, :, :], size=flownetsd_flow.size()[-2:], mode='nearest')
        diff_flownetsd_img1 = self.channelnorm(
            (x[:, :3, :, :]-diff_flownetsd_flow))

        # concat img1 flownetsd, flownets2, norm_flownetsd, norm_flownets2, diff_flownetsd_img1, diff_flownets2_img1
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow, norm_flownetsd_flow,
                            norm_flownets2_flow, diff_flownetsd_img1, diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)

        return flownetfusion_flow
