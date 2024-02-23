from mmdet.models.backbones.cbnet import CBRes2Net
# from mmdet.models.backbones.res2net import Res2Net
from mmdet.models.backbones import TridentResNet
from mmdet.models.necks.fpn import FPN

import torch
inputs = torch.rand(1, 3, 640, 640)  # 创建一个随机输入张量

model = TridentResNet(
        depth=101,
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=-1,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(2, ),)
outputs = model(inputs)  # 通过输入获取模型输出
print(type(outputs))
for level_out in outputs:
    print(level_out.shape)  # 打印各层级的形状


# neck_fpn = FPN(in_channels=[512, 1024, 2048,4096],out_channels=256,num_outs=5)
# fpn_out = neck_fpn(outputs)

