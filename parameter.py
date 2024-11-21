# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

from Best_module.VMLL import net
model = net()
model = model.cuda()
model.eval()
dummy_input = torch.randn(1, 3, 256, 256).cuda()
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f ' % (flops / 1000000.0, params))
