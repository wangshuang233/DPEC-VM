import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from Best_module.VMLL import unet
ckpts_dir = './weight/MUX_v1.pth'
net = unet()
net = net.cuda()
net.load_state_dict(torch.load(ckpts_dir), strict=True)
# net = nn.DataParallel(net)
net.eval()
t_all = 0
with torch.inference_mode():
    for i in range(10):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_ = torch.randn((1, 3, 3840, 2160)).cuda()
        t0 = time.time()
        prediction = net(input_)
        t1 = time.time()
        print("===> Processing time", (t1 - t0))
    for i in range(500):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_ = torch.randn((1, 3, 3840, 2160)).cuda()
        t0 = time.time()
        prediction = net(input_)
        t1 = time.time()
        t_all = t_all + (t1-t0)
        print("===> Processing time", (t1 - t0))
    print("averge time:",t_all/500)

