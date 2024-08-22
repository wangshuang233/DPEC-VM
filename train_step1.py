import argparse
import itertools
import time
from os import listdir
from os.path import join
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import lib.pytorch_ssim as pytorch_ssim
# from loss_ssim import SSIMa
from lib.data import get_training_set, is_image_file
from lib.utils import TVLoss, print_network, VGGPerceptualLoss
from torch.utils.tensorboard import SummaryWriter
from Best_module.BEE import net
import warnings
from os.path import join, isfile

# from interfence_seg import seg_main,parse_args
warnings.filterwarnings('ignore')
import logging
import os
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logging.basicConfig(level=logging.WARNING)
# Using model_LOL by default


masktensor_dict = {}


def histogram_loss(tensor1, tensor2):
    # 将张量移回到主机内存，并且不要求梯度
    tensor1_cpu = tensor1.detach().cpu().numpy()
    tensor2_cpu = tensor2.detach().cpu().numpy()
    # 将张量展平为一维数组
    flat_tensor1 = tensor1_cpu.flatten()
    flat_tensor2 = tensor2_cpu.flatten()

    # 计算直方图
    hist_tensor1, _ = np.histogram(flat_tensor1, bins=256, range=[0, 1])
    hist_tensor2, _ = np.histogram(flat_tensor2, bins=256, range=[0, 1])

    # 归一化直方图
    hist_tensor1 = hist_tensor1 / np.sum(hist_tensor1)
    hist_tensor2 = hist_tensor2 / np.sum(hist_tensor2)

    # 计算直方图差异度
    loss = 0.5 * np.sum(np.abs(hist_tensor1 - hist_tensor2))

    return loss


def cfg():
    parser = argparse.ArgumentParser(description='low-light image enhancement by SMNet')
    parser.add_argument('--trainset', type=str, default='./datasets/LOLv1_en/train', help='location of trainset')
    parser.add_argument('--testset', type=str, default='./datasets/LOLv1_en/test', help='location of testset')
    parser.add_argument('--output', default='output', help='location to save output images')
    parser.add_argument('--modelname', default='VMLL_LOLv1_pre', help='define model name')
    parser.add_argument('--deviceid', default="0", help='selecte which gpu device')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument('--lr_min', type=float, default=5e-5, help='Minimum Learning Rate')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=600, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=1, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=24, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped LR image')
    opt = parser.parse_args()
    return opt


def checkpoint(model, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_out_path = os.path.join(save_folder, "best.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def checkpoint_last(model, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    model_out_path = os.path.join(save_folder, "last.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def log_metrics(logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


best_psnr = 0

import shutil
import os


def copy_files_to_destination(file_list, destination_folder):
    for file_path in file_list:
        try:
            # 构建目标文件路径
            destination_path = os.path.join(destination_folder, os.path.basename(file_path))

            # 复制文件
            shutil.copy2(file_path, destination_path)
        except Exception as e:
            print(f"错误: {file_path} - {str(e)}")


def delete_files(file_list):
    for file_path in file_list:
        try:
            # 删除文件
            os.remove(file_path)
        except OSError as e:
            print(f"错误: {file_path} - {e.strerror}")


def eval(model, epoch, writer, txt_write, opt):
    print("==> Start testing")
    device = torch.device('cuda:' + opt.deviceid)
    tStart = time.time()
    trans = transforms.Compose([
        ToTensor()
    ])
    channel_swap = (1, 2, 0)
    model.eval()
    # Pay attention to the data structure
    test_LL_folder = os.path.join(opt.testset, "low")
    test_NL_folder = os.path.join(opt.testset, "high")
    test_est_folder = os.path.join(opt.output, opt.modelname, 'last')
    test_est_folder_best = os.path.join(opt.output, opt.modelname, 'best')

    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    try:
        os.stat(test_est_folder_best)
    except:
        os.makedirs(test_est_folder_best)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            LL_images = Image.open(test_LL_list[i]).convert('RGB')
            width, height = LL_images.size

            # 计算新的宽度和高度，使其都是8的倍数
            new_width = width - (width % 8)
            new_height = height - (height % 8)

            # 调整图像大小
            resized_LL_image = LL_images.resize((new_width, new_height))
            img_in = trans(resized_LL_image)

            LL_tensor = img_in.unsqueeze(0).to(device)

            # 使用模型进行推理
            prediction = model(LL_tensor)
           # 如果需要，进行预测结果的后处理
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

            # 将NumPy数组转换回Pillow图像
            est_image = Image.fromarray(prediction)

            # 将调整后的预测结果图像调整回原始尺寸
            resized_est_image = est_image.resize((width, height), resample=Image.BILINEAR)

            # 保存调整后的预测结果图像
            resized_est_image.save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, channel_axis=-1)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    save_folder = os.path.join('models', opt.modelname)
    file_checkpoint_last = checkpoint_last(model, save_folder)
    global best_psnr
    if best_psnr > psnr_score:
        delete_files(est_list)
    else:
        file_checkpoint = checkpoint(model, save_folder)
        best_psnr = psnr_score
        copy_files_to_destination(est_list, test_est_folder_best)
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    print("best_psnr:", best_psnr)
    writer.add_scalar('psnr', psnr_score, epoch)
    writer.add_scalar('ssim', ssim_score, epoch)
    txt_write.write(
        'EPOCH: ' + str(epoch) + ',' + ' PSNR ' + str(psnr_score)[:6] + ',' + ' SSIM ' + str(ssim_score)[:5] + '\n')
    return psnr_score, ssim_score


def logging(dircname):
    if not os.path.exists(dircname):
        os.makedirs(dircname)
    writer = SummaryWriter(dircname, opt.modelname)

    return writer


def C_DIVIDE_A(tensor1, tesnor2):
    C = tensor1.cpu()
    A = tesnor2.cpu()
    c_np = C.detach().numpy()
    a_np = A.detach().numpy()
    img_ret1 = cv2.divide(a_np, c_np)
    img_ret2 = cv2.divide(c_np, a_np)
    device = torch.device('cuda:' + opt.deviceid)
    return torch.from_numpy(img_ret1).to(device), torch.from_numpy(img_ret2).to(device)





def main(opt):
    writer = logging(os.path.join('tensorboard', opt.modelname))
    txt_name = 'metrics_' + opt.modelname + '.txt'
    with open(txt_name, mode='w') as txt_write:
        cuda = opt.gpu_mode
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        setup_seed(opt.seed)
        if cuda:
            # torch.cuda.manual_seed(opt.seed)
            cudnn.benchmark = True
        gpus_list = range(opt.gpus)

        # =============================#
        #   Prepare training data      #
        # =============================#
        print('===> Prepare training data')
        print('#### Now dataset is LOL ####')
        train_set = get_training_set(opt.trainset, 1, opt.patch_size, True)
        training_data_loader = DataLoader(dataset=train_set, num_workers=30, batch_size=opt.batchSize,
                                          pin_memory=True, shuffle=True, drop_last=True)
        # =============================#
        #          Build model         #
        # =============================#
        print('===> Build model')
        lighten = net()


        device = torch.device('cuda:' + opt.deviceid)
        lighten.to(device)
        print('---------- Networks architecture -------------')
        print_network(lighten)
        print('----------------------------------------------')

        # =============================#
        #         Loss function        #
        # =============================#
        L1_criterion = nn.L1Loss()
        TV_loss = TVLoss()
        mse_loss = torch.nn.MSELoss()
        ssim = pytorch_ssim.SSIM()
        percep_loss = VGGPerceptualLoss()
        smooth_criterion = nn.SmoothL1Loss()
        if cuda:
            gpus_list = range(opt.gpus)
            mse_loss = mse_loss.to(device)
            L1_criterion = L1_criterion.to(device)
            TV_loss = TV_loss.to(device)
            ssim = ssim.to(device)
            percep_loss = percep_loss.to(device)
            smooth_criterion = smooth_criterion.to(device)

        # =============================#
        #         Optimizer            #
        # =============================#
        parameters = [lighten.parameters()]
        # 使用 Adam 优化器
        optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

        # =============================#
        #         Training             #
        # =============================#

        for epoch in range(opt.start_iter, opt.nEpochs + 1):
            print('===> training epoch %d' % epoch)
            epoch_loss = 0
            lighten.train()

            tStart_epoch = time.time()
            for iteration, batch in enumerate(training_data_loader, 1):
                over_Iter = epoch * len(training_data_loader) + iteration
                optimizer.zero_grad()
                LL_t, NL_t = batch[0], batch[1]
                if cuda:
                    LL_t = LL_t.to(device)
                    NL_t = NL_t.to(device)
                t0 = time.time()
                LL_t_flatten = torch.flatten(LL_t)
                pred_t = lighten(LL_t)
                ori, ori_1 = C_DIVIDE_A(NL_t, LL_t)
                label, label_1 = C_DIVIDE_A(pred_t, LL_t)

                his_loss = histogram_loss(pred_t, NL_t)
                histogram_loss_B = histogram_loss(ori, label)


                pred_t_flatten = torch.flatten(pred_t)

                inner_loss = torch.dot(pred_t_flatten, LL_t_flatten) / (LL_t.shape[0]) / (LL_t.shape[1]) / (
                    LL_t.shape[2]) / (LL_t.shape[3])
                ssim_loss = 1 - ssim(pred_t, NL_t)
                tv_loss = TV_loss(pred_t)
                p_loss = percep_loss(pred_t, NL_t)
                smoothloss = smooth_criterion(pred_t, NL_t)
                loss = 2 * ssim_loss + 1.2 * p_loss + 0.8 * smoothloss + 1 * inner_loss + 0.01 * tv_loss + 1 * his_loss + 1 * histogram_loss_B  # 2024 1 24 1404 PSNR 20.09 0.8348
                # loss = 2 * ssim_loss + 2 * p_loss + 0.8 * smoothloss + 1 * inner_loss + 0.01 * tv_loss + 1 * his_loss + 1 * histogram_loss_B
                writer.add_scalar('ssim_loss', ssim_loss, over_Iter)
                writer.add_scalar('tv_loss', tv_loss * 0.01, over_Iter)
                writer.add_scalar('perceptual_loss', p_loss, over_Iter)
                writer.add_scalar('smooth_loss', smoothloss, over_Iter)
                writer.add_scalar('inner_loss', inner_loss, over_Iter)
                writer.add_scalar('histogram_loss_B', histogram_loss_B, over_Iter)
                loss.backward()
                optimizer.step()
                t1 = time.time()
                epoch_loss += loss
                if iteration % 10 == 0:
                    print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                          end=" ==> ")
                    logs = {
                        "loss": loss.data,
                        "ssim_loss": ssim_loss.data,
                        "tv_loss": tv_loss.data,
                        "percep_loss": p_loss.data,
                        "smooth_loss": smoothloss,
                        "inner_loss": inner_loss,
                        "histogram_loss_B": histogram_loss_B,
                    }
                    log_metrics(logs, over_Iter)
                    print("time: {:.4f} s".format(t1 - t0))

            writer.add_scalar('epoch_loss', float(epoch_loss / len(training_data_loader)), epoch)
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
                training_data_loader), time.time() - tStart_epoch))
            if epoch % (opt.snapshots) == 0:
                psnr_score, ssim_score = eval(lighten, epoch, writer, txt_write, opt)
                logs = {
                    "psnr": psnr_score,
                    "ssim": ssim_score,
                }
                log_metrics(logs, epoch, end_str="\n")

            # 使用余弦退火更新学习率，并设置最小学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr_min + (opt.lr - opt.lr_min) * (
                        1 + np.cos(np.pi * epoch / opt.nEpochs)) / 2
                print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
        print("======>>>Finished time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    opt = cfg()
    main(opt)