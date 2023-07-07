import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import timm

import time
import numpy as np
import cv2
from PIL import Image
import sklearn.calibration
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from tqdm import tqdm
import argparse

from util.util import *
from stats.classification import *
from dataset.basic_classification import *


class setup_config():
    def __init__(self):
        self.initilized = False
    
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to data')
        parser.add_argument('--csvpath', type=str, default=None, help='path to label dataframe(.csv)')
        parser.add_argument('--savepoint', required=True, help='path to save results')
        parser.add_argument('--name', type=str, default=None, help='save results to savepoint/name')
        ##data##
        parser.add_argument('--resize_factor', type=int, default=256, help='model input size')
        parser.add_argument('--gray_scale', action='store_true', help='grayscale')
        parser.add_argument('--fname_col', type=str, required=True, help='Dataframe column: filename')
        parser.add_argument('--label_col', type=str, required=True, help='Dataframe columns: labels')
        parser.add_argument('--class_name', nargs="+", default=["Normal", "Abnormal"], help='list of class')
        parser.add_argument('--gamma', action='store_true', help='gamma correction(gamma=0.7)')
        parser.add_argument('--clahe', action='store_true', help='CLAHE')
        ##model##
        parser.add_argument('--model', type=str, default='resnet152', help='classification model(timm)')
        parser.add_argument('--weights', type=str, required=True, help='path to model weights')
        ##optimize##
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        ##Grad-CAM##
        parser.add_argument('--gc_target', type=str, default='Calcification', help='target')
        parser.add_argument('--n_rank', type=int, default=5, help='N rank')
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


def test_acc(testloader, model, n_class, threshold=0.5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    correct = 0
    total = 0
    output_arr = np.ones((1, n_class))
    label_arr = np.array([])
    pred_arr = np.array([])
    model.cuda()
    model.eval()
   

    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # argmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            output_arr = np.concatenate((output_arr, outputs.softmax(1).cpu().numpy()), axis=0)
            label_arr = np.concatenate((label_arr, labels.cpu().numpy()), axis=0)
            pred_arr = np.concatenate((pred_arr, predicted.cpu().numpy()), axis=0)

    output_arr = np.delete(output_arr, 0, axis=0)
    acc = correct/total
    print('Accuracy on the test images: ', (100*correct/total))
    return acc, output_arr, label_arr, pred_arr


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def run(opt):
    assert os.path.exists(opt.dataroot)
    
    savepoint = os.path.join(opt.savepoint, opt.name)
    gen_new_dir(savepoint)
    
    if opt.gamma == True:
        img_loader_func = img_loader_gamma
    elif opt.clahe == True:
        img_loader_func = img_loader_clahe
    elif opt.gray_scale == True:
        img_loader_func = img_loader_grayscale
    else:
        img_loader_func = img_loader
    
    # Datasets
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
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False,
                                            pin_memory=(torch.cuda.is_available()), num_workers=8)
    
    # Models
    if opt.model == 'resnet50':
        model = timm.create_model('resnet50d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'resnet152':
        model = timm.create_model('resnet152d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'resnet200':
        model = timm.create_model('resnet200d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'densenet121':
        model = timm.create_model('densenet121', pretrained=True)
        if opt.gray_scale:
            model.features.conv0=torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'densenet169':
        model = timm.create_model('densenet169', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True)
        if opt.gray_scale:
            model.conv_stem=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'efficientnet_b4':
        model = timm.create_model('efficientnet_b4', pretrained=True)
        if opt.gray_scale:
            model.conv_stem=torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.classifier.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'convnext_small':
        model = timm.create_model('convnext_small', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'convnext_base':
        model = timm.create_model('convnext_base', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'convnext_large':
        model = timm.create_model('convnext_large', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=len(opt.class_name), bias=True)
        )    
    elif opt.model == 'convnextv2_small':
        model = timm.create_model('convnextv2_small', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    elif opt.model == 'convnextv2_base':
        model = timm.create_model('convnextv2_base', pretrained=True)
        if opt.gray_scale:
            model.stem[0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.head.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    else:
        model = timm.create_model('resnet152d', pretrained=True)
        if opt.gray_scale:
            model.conv1[0]=torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        model = torch.nn.Sequential(
            model,
            torch.nn.Linear(model.fc.out_features, out_features=len(opt.class_name), bias=True)
        )
    
    test_model = model
    test_model.load_state_dict(torch.load(opt.weights))
    
    acc, output_arr, label_arr, pred_arr = test_acc(test_loader, test_model, n_class=len(opt.class_name))
    
    # Result Statistics
    print(metrics.classification_report(label_arr, pred_arr))
    
    statistic_savepath = os.path.join(savepoint, 'statistics')
    gen_new_dir(statistic_savepath)
    
    if len(opt.class_name) > 2:
        test_fname_arr = np.array([os.path.splitext(os.path.basename(p))[0] for p in test_dataset.data_path])
        conf_df = pd.DataFrame(np.concatenate([np.expand_dims(test_fname_arr,1), output_arr, np.expand_dims(test_dataset.labels,1)], axis=1), columns=['FileName', *opt.class_name, 'Label'])
        conf_df.to_csv(os.path.join(statistic_savepath, 'test_results.csv'), index=False)
        
        cf_matrix = metrics.confusion_matrix(label_arr, pred_arr)
        
        fig, ax = plt.subplots(1, 2, figsize=(20,8))

        sns.heatmap(cf_matrix, 
                    xticklabels=opt.class_name, 
                    yticklabels=opt.class_name, 
                    annot=True, fmt = 'd', cmap='Blues', ax=ax[0])
        
        ax[0].set_title('Confusion Matrix', size = 15)
        ax[0].set_xlabel('Predicted class', size = 13)
        ax[0].set_ylabel('Actual class', size = 13) 
        
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "darkgreen", "blueviolet", "olive", "lightpink"])
        label_arr_bin = label_binarize(label_arr, classes=list(range(len(opt.class_name))))
        n_classes = len(opt.class_name)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_arr_bin[:, i], output_arr[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(label_arr_bin.ravel(), output_arr.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        ax[1].plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.4f})".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        ax[1].plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.4f})".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        for i, color in zip(range(n_classes), colors):
            ax[1].plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.4f})".format(i, roc_auc[i]),
            )


        ax[1].plot([0, 1], [0, 1], "k--", lw=2)
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].set_title("Receiver operating characteristic to multiclass")
        ax[1].legend(loc="lower right")
        
        plt.savefig(os.path.join(statistic_savepath, 'confusion_matrix_auc.png'), dpi=72)
        plt.close(fig)

    else:
        # Youden’s J statistic
        fpr, tpr, thresholds = metrics.roc_curve(label_arr, output_arr[:, 1])
        J = tpr - fpr 
        idx = np.argmax(J)
        best_thresh = thresholds[idx]
        roc_auc = metrics.auc(fpr, tpr)
        sens, spec = tpr[idx], 1-fpr[idx]

        with open(os.path.join(statistic_savepath, 'YoudensJstatistic.txt'), "w") as file:
            file.write(f"ROCAUC: {roc_auc}\n")
            file.write(f"Best threshold(Youden’s J statistic): {best_thresh}\n")
            file.write(f"Sensitivity: {sens}\n")
            file.write(f"Specificity: {spec}\n")
            file.close()

        test_fname_arr = np.array([os.path.splitext(os.path.basename(p))[0] for p in test_dataset.data_path])
        conf_df = pd.DataFrame(np.stack([test_fname_arr, output_arr[:,1], test_dataset.labels], axis=1), columns=['FileName', 'Pred', 'Label'])
        conf_df.to_csv(os.path.join(statistic_savepath, 'test_results.csv'), index=False)

        tn, fp, fn, tp = metrics.confusion_matrix(label_arr, pred_arr).ravel()

        fig, ax = plt.subplots(1, 2, figsize=(20,8))

        sns.heatmap([[tp, fp],[fn, tn]], 
                    xticklabels=opt.class_name[::-1], 
                    yticklabels=opt.class_name[::-1], 
                    annot=True, fmt = 'd', cmap='Blues', ax=ax[0])

        ax[0].set_title('Confusion Matrix', size = 15)
        ax[0].set_xlabel('Actual class', size = 13)
        ax[0].set_ylabel('Predicted class', size = 13) 

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(opt.class_name)):
            fpr[i], tpr[i], _ = metrics.roc_curve(label_arr==i, output_arr[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        lw = 2
        ax[1].plot(
            fpr[1],
            tpr[1],
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.4f)" % roc_auc[1],
        )
        ax[1].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].set_title("Receiver operating characteristic")
        ax[1].legend(loc="lower right")

        plt.savefig(os.path.join(statistic_savepath, 'confusion_matrix_auc.png'), dpi=72)
        plt.close(fig)

        fraction_of_positives, mean_predicted_value = sklearn.calibration.calibration_curve(label_arr, output_arr[:, 1], n_bins=10)
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        ax1.plot([0, 1], [0, 1], "k:")
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-")
        ax2.hist(output_arr[:,1], range=(0, 1), bins=10, histtype="step", lw=2)
        ax1.set_title("Calibration plots")
        ax1.set_ylabel("Fraction of positives")
        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")

        plt.savefig(os.path.join(statistic_savepath, 'calibration_plots.png'), dpi=72)
        plt.close(fig)

        thre_sen_spe_df = get_classification_stat_basedon_threshold(label_arr, output_arr[:,1], interval=.05)
        thre_sen_spe_df.to_csv(os.path.join(statistic_savepath, 'sensitivity_and_specificity.csv'), index=False)

        print(f'AUC: {roc_auc[1]}')
#     print('Extracting GradCAM...')
#     # Grad-CAM
#     test_dataset_vis = CACCalcifDataset(data_dir=opt.dataroot, 
#                                         csv_path=opt.csvpath, 
#                                         resize_factor=opt.resize_factor,
#                                         gray_scale=opt.gray_scale,
#                                         fname_col=opt.fname_col,
#                                         label_col=opt.label_col,
#                                         fold=opt.fold,
#                                         transform=get_transform,
#                                         mode='test', 
#                                         convert=False)
    
#     test_result_df = pd.DataFrame(np.array(test_dataset.data_path), columns=['Filedir'])
#     test_result_df = pd.concat([test_result_df, 
#                                 pd.DataFrame(np.array(output_arr), columns=['Conf_'+ str(i) for i in range(len(opt.class_name))]),
#                                 pd.DataFrame(np.array(pred_arr), columns=['Pred']),
#                                 pd.DataFrame(np.array(label_arr), columns=['Label'])],axis=1)
#     test_result_df = test_result_df.astype({'Pred':np.uint8, 
#                                             'Label':np.uint8})
    
#     LABEL_INT2STR = {i:name for i, name in enumerate(opt.class_name)}
#     target_idx = opt.class_name.index('Calcification')
    
#     if opt.model.startswith('resnet'):
#         target_layer = [test_model.layer4[-1]]
#     elif opt.model.startswith('densenet'):
#         target_layer = [test_model.features[-1]]
#     else:
#         pass
#     cam = GradCAM(model=test_model, target_layers=target_layer, use_cuda=True)
    
#     #Top-N FalsePositive
#     print(f'Top-{opt.n_rank} FalsePositive GradCAM...')
#     fp_savepath = os.path.join(savepoint, 'GradCAM', 'FalsePositive')
#     gen_new_dir(fp_savepath)

#     FP_idx_ls = list(test_result_df[test_result_df.Label != target_idx].sort_values(by=f'Conf_{target_idx}',ascending=False).index)
    
#     rank_cnt = 0
#     for i in FP_idx_ls[:opt.n_rank]:
#         feature, label = test_dataset[i]
#         img, _ = test_dataset_vis[i]
#         feature = torch.unsqueeze(feature, 0)
#         img = img.astype(np.float)/255

#         class_cam_image = []

#         for l in range(len(LABEL_INT2STR)):
#             class_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(l)],aug_smooth=True,eigen_smooth=True)
#             class_cam = class_cam[0, :]
#             class_cam_image.append(show_cam_on_image(img, class_cam, use_rgb=True))

#         prob = test_model(feature.cuda()).softmax(1)[0].cpu().data.numpy()
#         #print(test_model(features.cuda()).softmax(1)[0][1], labels)
#         fig,ax = plt.subplots(1, len(LABEL_INT2STR)+1, figsize=(20, 15))
#         ax[0].imshow(img)
#         ax[0].set_title(f'original image({LABEL_INT2STR[label]})')
#         for l in range(len(LABEL_INT2STR)):
#             ax[l+1].imshow(class_cam_image[l])
#             ax[l+1].set_title(f'{LABEL_INT2STR[l]}(Conf:{prob[l]:.2f})')

#         plt.suptitle(f'{test_dataset.data_path[i]} | ground truth: {label}',y=.73) 

#         plt.savefig(os.path.join(fp_savepath, f'FP_{str(rank_cnt).zfill(3)}.png'), dpi=72)
#         plt.close(fig)

#         rank_cnt += 1
    
#     #Top-N TrueNegative
#     print(f'Top-{opt.n_rank} TrueNegative GradCAM...')
#     tn_savepath = os.path.join(savepoint, 'GradCAM', 'TrueNegative')
#     gen_new_dir(tn_savepath)

#     TN_idx_ls = list(test_result_df[test_result_df.Label != target_idx].sort_values(by=f'Conf_{target_idx}',ascending=True).index)

#     rank_cnt = 0
#     for i in TN_idx_ls[:opt.n_rank]:
#         feature, label = test_dataset[i]
#         img, _ = test_dataset_vis[i]
#         feature = torch.unsqueeze(feature, 0)
#         img = img.astype(np.float)/255

#         class_cam_image = []

#         for l in range(len(LABEL_INT2STR)):
#             class_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(l)],aug_smooth=True,eigen_smooth=True)
#             class_cam = class_cam[0, :]
#             class_cam_image.append(show_cam_on_image(img, class_cam, use_rgb=True))

#         prob = test_model(feature.cuda()).softmax(1)[0].cpu().data.numpy()
#         #print(test_model(features.cuda()).softmax(1)[0][1], labels)
#         fig,ax = plt.subplots(1, len(LABEL_INT2STR)+1, figsize=(20, 15))
#         ax[0].imshow(img)
#         ax[0].set_title(f'original image({LABEL_INT2STR[label]})')
#         for l in range(len(LABEL_INT2STR)):
#             ax[l+1].imshow(class_cam_image[l])
#             ax[l+1].set_title(f'{LABEL_INT2STR[l]}(Conf:{prob[l]:.2f})')

#         plt.suptitle(f'{test_dataset.data_path[i]} | ground truth: {label}',y=.73) 

#         plt.savefig(os.path.join(tn_savepath, f'TN_{str(rank_cnt).zfill(3)}.png'), dpi=72)
#         plt.close(fig)
        
#         rank_cnt += 1
    
#     #Top-N FalseNegative
#     print(f'Top-{opt.n_rank} FalseNegative GradCAM...')
#     fn_savepath = os.path.join(savepoint, 'GradCAM', 'FalseNegative')
#     gen_new_dir(fn_savepath)

#     FN_idx_ls = list(test_result_df[test_result_df.Label == target_idx].sort_values(by=f'Conf_{target_idx}',ascending=True).index)
    
#     rank_cnt = 0
#     for i in FN_idx_ls[:opt.n_rank]:
#         feature, label = test_dataset[i]
#         img, _ = test_dataset_vis[i]
#         feature = torch.unsqueeze(feature, 0)
#         img = img.astype(np.float)/255

#         class_cam_image = []

#         for l in range(len(LABEL_INT2STR)):
#             class_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(l)],aug_smooth=True,eigen_smooth=True)
#             class_cam = class_cam[0, :]
#             class_cam_image.append(show_cam_on_image(img, class_cam, use_rgb=True))

#         prob = test_model(feature.cuda()).softmax(1)[0].cpu().data.numpy()
#         #print(test_model(features.cuda()).softmax(1)[0][1], labels)
#         fig,ax = plt.subplots(1, len(LABEL_INT2STR)+1, figsize=(20, 15))
#         ax[0].imshow(img)
#         ax[0].set_title(f'original image({LABEL_INT2STR[label]})')
#         for l in range(len(LABEL_INT2STR)):
#             ax[l+1].imshow(class_cam_image[l])
#             ax[l+1].set_title(f'{LABEL_INT2STR[l]}(Conf:{prob[l]:.2f})')

#         plt.suptitle(f'{test_dataset.data_path[i]} | ground truth: {label}',y=.73) 

#         plt.savefig(os.path.join(fn_savepath, f'FN_{str(rank_cnt).zfill(3)}.png'), dpi=72)
#         plt.close(fig)

#         rank_cnt += 1
    
#     #Top-N TruePositive
#     print(f'Top-{opt.n_rank} TruePositive GradCAM...')
#     tp_savepath = os.path.join(savepoint, 'GradCAM', 'TruePositive')
#     gen_new_dir(tp_savepath)

#     TP_idx_ls = list(test_result_df[test_result_df.Label == target_idx].sort_values(by=f'Conf_{target_idx}',ascending=False).index)

#     rank_cnt = 0
#     for i in TP_idx_ls[:opt.n_rank]:
#         feature, label = test_dataset[i]
#         img, _ = test_dataset_vis[i]
#         feature = torch.unsqueeze(feature, 0)
#         img = img.astype(np.float)/255

#         class_cam_image = []

#         for l in range(len(LABEL_INT2STR)):
#             class_cam = cam(input_tensor=feature, targets=[ClassifierOutputTarget(l)],aug_smooth=True,eigen_smooth=True)
#             class_cam = class_cam[0, :]
#             class_cam_image.append(show_cam_on_image(img, class_cam, use_rgb=True))

#         prob = test_model(feature.cuda()).softmax(1)[0].cpu().data.numpy()
#         #print(test_model(features.cuda()).softmax(1)[0][1], labels)
#         fig,ax = plt.subplots(1, len(LABEL_INT2STR)+1, figsize=(20, 15))
#         ax[0].imshow(img)
#         ax[0].set_title(f'original image({LABEL_INT2STR[label]})')
#         for l in range(len(LABEL_INT2STR)):
#             ax[l+1].imshow(class_cam_image[l])
#             ax[l+1].set_title(f'{LABEL_INT2STR[l]}(Conf:{prob[l]:.2f})')

#         plt.suptitle(f'{test_dataset.data_path[i]} | ground truth: {label}',y=.73) 

#         plt.savefig(os.path.join(tp_savepath, f'TP_{str(rank_cnt).zfill(3)}.png'), dpi=72)
#         plt.close(fig)

#         rank_cnt += 1
    
    print('Validation done!')
    
    
if __name__=='__main__':
    opt = setup_config().parse()
    run(opt)
