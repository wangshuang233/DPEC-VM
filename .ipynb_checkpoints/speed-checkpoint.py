import torch
# from mamba_unet import unet as net
# from model_LOL_ini import lowlightnet3 as net
from Retinex_mamba import RetinexMamba as net
iterations = 500   # 重复计算的轮次

model = net()
device = torch.device("cuda:0")
model.to(device)

random_input = torch.randn(1, 3, 600, 400).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(random_input)

# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(random_input)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
