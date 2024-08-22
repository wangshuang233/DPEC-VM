import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from lib.dataset import is_image_file
from PIL import Image
from os import listdir
import os


def eval(opt):
    # Define gpu device
    # device = torch.device('cuda:{}'.format(opt.device))
    device = torch.device("cuda")
    # Load model
    model = unet()
    model = model.to(device)

    model.load_state_dict(torch.load(opt.modelfile))
    model.eval()

    # Get filename; Please ensure  both h&w resolution of inpu image can devided by 4, such as 600*400
    LL_filename = os.path.join(opt.test_folder)
    est_filename = os.path.join(opt.output)
    try:
        os.stat(est_filename)
    except:
        os.mkdir(est_filename)
    LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename))]

    print(LL_filename)
    Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename))]
    print(Est_img)
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    time_ave = 0

    for i in range(LL_image.__len__()):
    # for i in range(50):
        with torch.no_grad():
            LL_in = Image.open(LL_image[i]).convert('RGB')
            width, height = LL_in.size

            # 计算新的宽度和高度，使其都是8的倍数
            new_width = width - width%8
            new_height = height - height%8
            if new_width > 3840:
                new_width = 3840
            if new_height > 2160:
                new_height = 2160
            resized_LL_image = LL_in.resize((new_width, new_height))
            img_in = trans(resized_LL_image)
            LL_tensor = img_in.unsqueeze(0).to(device)

            t0 = time.time()
            prediction = model(LL_tensor)
            t1 = time.time()
            time_ave += (t1 - t0)

            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)
            prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

            # 将NumPy数组转换回Pillow图像
            est_image = Image.fromarray(prediction)

            # 将调整后的预测结果图像调整回原始尺寸
            resized_est_image = est_image.resize((width, height), resample=Image.BILINEAR)

            # 保存调整后的预测结果图像
            resized_est_image.save(Est_img[i])

            print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='low-light image enhancement by SMNet')
    parser.add_argument('--test_folder', type=str, default='./datasets/LSRW/test/low',
                        help='location to input images')
    parser.add_argument('--output', default='./test_out', help='location to save output images')
    parser.add_argument('--device', type=str, default='0')

    # modelfile and modeltype should be same kind
    parser.add_argument('--modelfile', default='./weight/best_LSRW.pth',
                        help='pretrained model LOL or sdsd_in')
    parser.add_argument('--modeltype', type=str, default='low',
                        help="to choose pretrained model training on LOL or sdsd_in")

    parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
    parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

    opt = parser.parse_args()
    print(opt)

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found!!")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    # To choose pretrained model training on LOL or sdsd_in
    if str.lower(opt.modeltype) == 'sdsd_in':
        print('======>Now using sdsd_in')
        from ResVMUNetX import unet
    else:
        print("======>Now using default model_LOL")
        from ResVMUNetX import unet

    eval(opt)