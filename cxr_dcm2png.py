import numpy as np
import pandas as pd
import PIL.Image as Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import argparse

from util.util import *
from cxrtools.load_cxr import *

class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--data_dir', required=True, help='path to rsna mammography data')
        parser.add_argument('--resize_factor', type=int, default=512, help='resize factor')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        self.initialized = True
        return parser
    
    
    def gather_options(self):
        if not self.initilized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='Heart segmentation model inference')
            parser = self.initialize(parser)
            self.parser = parser
        return parser.parse_args()
    
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        # save to the disk
#         expr_dir = os.path.join(opt.checkpoint, opt.name)
#         gen_new_dir(expr_dir)
#         file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.name))
#         with open(file_name, 'wt') as opt_file:
#             opt_file.write(message)
#             opt_file.write('\n')
    
    
    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        
#         if opt.name is None:
#             opt.name = ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S'))
        
#         # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix
        
        self.print_options(opt)
        self.opt = opt
        return self.opt

def save_arr_to_png(arr, save_dir, fname):
    """save np.array to png file
    Parameters:
        arr (np.array) -- image array
        save_dir (str) -- directory to save array
        fname (str) -- file name
    """
    save_path = os.path.join(save_dir, fname+'.png')
    Image.fromarray(arr.astype(np.uint8)).save(save_path)

def run(opt):
    resize_factor=opt.resize_factor
    gen_new_dir(opt.savepoint)
    dcm_path_ls = load_file_path(opt.data_dir, DCM_EXTENSION)
    for file_name, file_path in tqdm([[os.path.splitext(os.path.basename(p))[0], p] for p in dcm_path_ls]):
        cxr_img = cxr_loader(file_path)
        
        h, w = cxr_img.shape
        resize_scale = resize_factor/max(w,h)
        resized_w = round(w*resize_scale)
        resized_h = round(h*resize_scale)
        cxr_img = cv2.resize(cxr_img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        
        save_arr_to_png(cxr_img, opt.savepoint, file_name)
    
if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
