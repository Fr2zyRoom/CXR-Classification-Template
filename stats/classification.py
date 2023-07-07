import numpy as np
import pandas as pd
from sklearn import metrics


def get_classification_stat_basedon_threshold(y_true, y_pred, interval=.05):
    """get sensitivity and specificity according to threshold change
    
    Parameters:
        y_true (array-like of shape (n_samples,)) -- ground truth
        y_pred (array-like of shape (n_samples,)) -- prediction(confidence value)
        interval(float) -- check all sub directories
    
    Return:
        file_paths (DataFrame) -- sensitivity and specificity according to threshold change 
    """
    threshold_ls = np.arange(0.,1.,interval)[1:]
    thre_sen_spe_ls = []
    for threshold in threshold_ls:
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred>threshold).ravel()
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        thre_sen_spe_ls.append([threshold, sensitivity, specificity])
        
    return pd.DataFrame(np.array(thre_sen_spe_ls), columns=['threshold','sensitivity','specificity'])
