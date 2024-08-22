from os.path import join
from lib.dataset import DatasetFromFolderEval, DatasetFromFolder, Lowlight_DatasetFromVOC,DatasetFromFolderLab
import torchvision.transforms as transforms


def transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, 'high')
    lr_dir = join(data_dir, 'low')
    return DatasetFromFolder(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                              transform=transform())

def get_training_set_lab(data_dir, upscale_factor, patch_size, data_augmentation):
    hr_dir = join(data_dir, 'high')
    lr_dir = join(data_dir, 'low')
    return DatasetFromFolderLab(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation,
                              transform=transform())

def get_eval_set(lr_dir, upscale_factor):
    return DatasetFromFolderEval(lr_dir, upscale_factor,
                             transform=transform())

def get_Low_light_training_set(upscale_factor, patch_size, data_augmentation):
    return Lowlight_DatasetFromVOC(patch_size, upscale_factor, data_augmentation,
                             transform=transform())

