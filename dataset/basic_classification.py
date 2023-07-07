import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure
from tqdm import tqdm

from util.util import *
from tools.img_tools import *

def img_loader(path):
    img = np.array(Image.open(path).convert('L'))
    return img


def img_loader_gamma(path):
    img = np.array(Image.open(path).convert('L'))
    return cosine_gamma_correction(img)


def img_loader_clahe(path):
    img = np.array(Image.open(path).convert('L'))
    return clahe(img)


def match_data_and_label(dataroot, 
                         meta_df,
                         fname_col=None,
                         label_col=None):
    """match data(dataroot) and labels(DataFrame)
    Parameters:
        dataroot (str) -- directory to data
        meta_df (DataFrame) -- meta data
        fname_col (str) -- a filename(data) column in the label DataFrame 
        label_col (str) -- label column in the label DataFrame 
        
    Return:
        data_label_ls (list) -- a list of data_path and labels(matched)
    """
    print('matching cxr image path and label...')
    data_label_ls=[]
    
    for filename, label in tqdm(meta_df[[fname_col, label_col]].values):
        filepath = os.path.join(dataroot,filename+'.png')
        if os.path.exists(filepath):
            data_label_ls.append([filepath,label])
    
    print('Done!')
    return data_label_ls


def gen_dataset_ls(dataroot, 
                   meta_df_path,
                   fname_col=None,
                   label_col=None,
                   split=None):
    """match data(dataroot) and labels(label_csv)
    Parameters:
        dataroot (str) -- directory to data
        meta_df_path (str) -- a path of meta data dataframe(.csv)
        split (str) -- if dataset need to split, put 'train'/'valid'/'test'
        
    Return:
        data_label_ls (list) -- a list of data_path and labels(matched)
    """
    meta_df = pd.read_csv(meta_df_path)
    if split:
        meta_df = meta_df[meta_df.split==split]
    data_label_ls = match_data_and_label(dataroot, meta_df, fname_col, label_col)
    return data_label_ls


def get_transform(params=None, mode=None, resize_factor=512, gray_scale=False, convert=True):
    transform_list = []
    #padding
    transform_list.append(
        A.augmentations.geometric.resize.LongestMaxSize(max_size=resize_factor, interpolation=cv2.INTER_AREA)
    )
    transform_list.append(
        A.augmentations.geometric.transforms.PadIfNeeded(min_height=resize_factor, 
                                                         min_width=resize_factor, 
                                                         border_mode=cv2.BORDER_CONSTANT, 
                                                         value=0)
    )
    if mode=='train':
        ## geometric transform
        transform_list.append(
            A.ShiftScaleRotate(shift_limit=0.03, 
                               scale_limit=0.1, 
                               rotate_limit=10, 
                               p=0.4, 
                               border_mode = cv2.BORDER_CONSTANT)
        )
        ## brightness or contrast
        transform_list.append(
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, 
                                           contrast_limit=0.1),
                A.RandomGamma(p=1)
            ], p=.3)
        )
        ## blur or sharpen
        transform_list.append(
            A.OneOf([
                A.GaussianBlur(sigma_limit=(0,0.1)),
                A.Sharpen(alpha=(0., 0.1))
            ], p=.2)
        )
        # # cutout
        # transform_list.append(
        #     A.augmentations.dropout.cutout.Cutout(num_holes=2, max_h_size=128, max_w_size=64, fill_value=0, always_apply=False, p=0.3)
        # )
    ## normalize
    if convert:
        if gray_scale:
            transform_list.append(
                A.Normalize(mean=[0.5,],
                            std=[0.5,],)
            )
        else:
            transform_list.append(
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],)
            )
        transform_list.append(ToTensorV2())
    
    return A.Compose(transform_list)


class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir, 
                 meta_df_path, 
                 fname_col=None,
                 label_col=None,
                 img_loader=img_loader,
                 resize_factor=512,
                 gray_scale=False,
                 transform=get_transform,
                 mode=None, 
                 convert=True):
        self.data_dir = data_dir
        self.meta_df_path = meta_df_path
        self.fname_col = fname_col
        self.label_col = label_col
        self.img_loader = img_loader
        self.gray_scale = gray_scale
        self.mode = mode
        
        if self.gray_scale:
            self.convert_func = lambda x: x
        else:
            self.convert_func = gray2rgb
        
        self.transform = transform(mode=self.mode, 
                                   resize_factor=resize_factor, 
                                   gray_scale=self.gray_scale,
                                   convert=convert)
        
        data_label_ls = gen_dataset_ls(self.data_dir, 
                                       self.meta_df_path,
                                       fname_col=self.fname_col,
                                       label_col=self.label_col,
                                       split=self.mode)
        
        self.data_path, self.labels = list(zip(*data_label_ls))
        
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        data = self.convert_func(self.img_loader(self.data_path[index]))
        if self.transform:
            data = self.transform(image=data)['image']
        #label = np.array(self.labels[index]).astype(float)
        
        return data, self.labels[index] #torch.FloatTensor(label)
    
    def get_labels(self):
        return self.labels