import os
import cv2
import torch
from PIL import Image
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np
import lpips
import torch
import torchvision
import torchvision.transforms as transforms

def count_files(directory):
    total_files = 0
    for entry in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, entry)):
            total_files += 1
    return total_files
def lpips_loss(img1, img2):
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    # 加载预训练的LPIPS模型
    lpips_model = lpips.LPIPS(net="vgg")

    # 将图像转换为PyTorch的Tensor格式
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # 使用LPIPS模型计算距离
    distance = lpips_model(image1_tensor, image2_tensor)
    return  distance.item()

test_NL_folder = "./datasets/LSRW/test/high"
test_LL_folder = "./output/MUX_LSRW_812/best"
lens = count_files(test_NL_folder)
print("the number of dataset:",lens)
psnr_score = 0
ssim_score = 0
lpips_score = 0
for filename in os.listdir(test_NL_folder):
    # 拼接完整的文件路径并添加到列表中
    gt_path = os.path.join(test_NL_folder, filename)
    # est_path = os.path.join(test_LL_folder, filename[:-3]+"png")
    est_path = os.path.join(test_LL_folder, filename)
    gt = cv2.imread(gt_path)
    est = cv2.imread(est_path)
    psnr_val = compare_psnr(gt, est, data_range=255)
    ssim_val = compare_ssim(gt, est, channel_axis=-1)
    lpips_val = lpips_loss(gt_path, est_path)
    psnr_score = psnr_score + psnr_val
    ssim_score = ssim_score + ssim_val
    lpips_score = lpips_score + lpips_val
psnr_score = psnr_score / lens
ssim_score = ssim_score / lens
lpips_score = lpips_score/ lens
print("psnr_score:", psnr_score)
print("ssim_score：", ssim_score)
print("lpips_val:", lpips_score)



