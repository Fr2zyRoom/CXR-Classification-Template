import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import timm

import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import argparse

from util.util import *
from util.train_template import *
from dataset.basic_classification import *
from loss.eql_loss import *
from loss.focal_loss import *


class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to data')
        parser.add_argument('--csvpath', required=True, help='path to label dataframe(.csv)')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        parser.add_argument('--wandb_project', required=True, help='project name')
        parser.add_argument('--name', type=str, default=None, help='save results to savepoint/wandb_project/name')
        parser.add_argument('--seed', type=int, default=4, help='fix random seed')
        ##data##
        parser.add_argument('--resize_factor', type=int, default=256, help='model input size')
        parser.add_argument('--gray_scale', action='store_true', help='grayscale')
        parser.add_argument('--fname_col', type=str, required=True, help='Dataframe column: filename')
        parser.add_argument('--label_col', type=str, required=True, help='Dataframe columns: labels')
        parser.add_argument('--imbalance', action='store_true', help='imbalance dataset')
        parser.add_argument('--class_num', type=int, required=True, help='Number of class')
        parser.add_argument('--gamma', action='store_true', help='gamma correction(gamma=0.7)')
        parser.add_argument('--clahe', action='store_true', help='CLAHE')
        ##model##
        parser.add_argument('--model', type=str, default='resnet152', help='classification model(timm)')
        ##optimize##
        parser.add_argument('--criterion', type=str, default='crossentropy', help='loss function')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--n_epochs', type=int, default=300, help='train epochs')
        parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
        parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
        ##print##
        parser.add_argument('--train_print_freq', type=int, default=100, help='print frequency(train)')
        parser.add_argument('--test_print_freq', type=int, default=10, help='print frequency(test)')
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
        
        if opt.name is None:
            opt.name = ''.join(datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime('%Y-%m-%d-%H-%M-%S'))
        
#         # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix
        
        self.print_options(opt)
        self.opt = opt
        return self.opt


def run(opt):
    assert os.path.exists(opt.dataroot)
    
    savepoint = os.path.join(opt.savepoint, opt.wandb_project, opt.name)
    gen_new_dir(savepoint)
    
    if opt.gamma == True:
        img_loader_func = img_loader_gamma
    elif opt.clahe == True:
        img_loader_func = img_loader_clahe
    else:
        img_loader_func = img_loader
    
    # Datasets
    train_dataset = CXRDataset(data_dir=opt.dataroot, 
                               meta_df_path=opt.csvpath, 
                               fname_col=opt.fname_col,
                               label_col=opt.label_col,
                               img_loader=img_loader_func,
                               resize_factor=opt.resize_factor,
                               gray_scale=opt.gray_scale,
                               transform=get_transform,
                               mode='train', 
                               convert=True)
    val_dataset = CXRDataset(data_dir=opt.dataroot, 
                             meta_df_path=opt.csvpath, 
                             fname_col=opt.fname_col,
                             label_col=opt.label_col,
                             img_loader=img_loader_func,
                             resize_factor=opt.resize_factor,
                             gray_scale=opt.gray_scale,
                             transform=get_transform,
                             mode='valid', 
                             convert=True)
    test_dataset = CXRDataset(data_dir=opt.dataroot, 
                              meta_df_path=opt.csvpath, 
                              fname_col=opt.fname_col,
                              label_col=opt.label_col,
                              img_loader=img_loader_func,
                              resize_factor=opt.resize_factor,
                              gray_scale=opt.gray_scale,
                              transform=get_transform,
                              mode='test', 
                              convert=True)
    
    # Models
    if opt.model == 'resnet50':
        model = timm.create_model('resnet50d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'resnet152':
        model = timm.create_model('resnet152d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'resnet200':
        model = timm.create_model('resnet200d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'densenet121':
        model = timm.create_model('densenet121', pretrained=True)
        if opt.gray_scale:
            model.features.conv0=torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'densenet169':
        model = timm.create_model('densenet169', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
        if opt.gray_scale:
            model.conv_stem=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', pretrained=True)
        if opt.gray_scale:
            model.conv_stem=torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'convnext_small':
        model = timm.create_model('convnext_small', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=opt.class_num, bias=True)
        )    
    elif opt.model == 'convnextv2_small':
        model = timm.create_model('convnextv2_small', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=opt.class_num, bias=True)
        )
    elif opt.model == 'convnextv2_base':
        model = timm.create_model('convnextv2_base', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=opt.class_num, bias=True)
        )
    else:
        model = timm.create_model('resnet152d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=opt.class_num, bias=True)
        )

    #print(model)
    # Criterions
    if opt.criterion == 'cross_entropy':
        criterion = torch.nn.functional.cross_entropy
    elif opt.criterion == 'focal':
        criterion = FocalLoss(gamma=2.)
    elif opt.criterion == 'alpha_focal':
        criterion = FocalLoss(torch.tensor([0.5, 1., 0.1]).cuda(), gamma=2.) #0.3, 0.5, 0.2
    elif opt.criterion == 'cross_entropy_softlabel':
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.25)
        #criterion = torch.nn.BCELoss
    elif opt.criterion == 'eql_loss':
        criterion = SoftmaxEQL(num_classes=2)
        #criterion = torch.nn.BCELoss
    else:
        criterion = torch.nn.functional.cross_entropy
    
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Train the model
    set_all_seeds(opt.seed)
    train(model=model, 
          criterion=criterion,
          train_set=train_dataset, 
          valid_set=val_dataset, 
          test_set=test_dataset, 
          save=savepoint,
          n_epochs=opt.n_epochs, 
          batch_size=opt.batch_size, 
          lr=opt.lr, 
          patience=opt.patience, 
          imbalance=opt.imbalance,
          train_print_freq=opt.train_print_freq, 
          test_print_freq=opt.test_print_freq,
          project_name=opt.wandb_project, 
          trial_name=opt.name)
    print('Model traing done!')
    
    
if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
