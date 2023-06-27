import torch
from sklearn.metrics import roc_curve, auc
import numpy as np

# from SMDD https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset/blob/main/utils.py

def get_apcer_op(apcer, bpcer, threshold, op):
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]

def get_bpcer_op(apcer, bpcer, threshold, op):
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]

def performances_compute(prediction_scores, gt_labels, threshold_type='eer', op_val=0.1, verbose=True, positive_label=1):
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=positive_label)
    bpcer = 1 - tpr
    val_eer, _, eer_threshold = get_eer_threhold(fpr, tpr, threshold)
    val_auc = auc(fpr, tpr)

    if threshold_type=='eer':
        threshold = eer_threshold
    elif threshold_type=='apcer':
        _, _, threshold = get_apcer_op(fpr, bpcer, threshold, op_val)
    elif threshold_type=='bpcer':
        _, _, threshold = get_bpcer_op(fpr, bpcer, threshold, op_val)
    else:
        threshold = 0.5

    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    threshold_APCER = type2 / num_fake
    threshold_BPCER = type1 / num_real
    threshold_ACER = (threshold_APCER + threshold_BPCER) / 2.0

    if verbose is True:
        print(f'AUC@ROC: {val_auc}, threshold:{threshold}, EER: {val_eer}, APCER:{threshold_APCER}, BPCER:{threshold_BPCER}, ACER:{threshold_ACER}')

    return val_auc, val_eer, threshold_APCER, threshold_BPCER, threshold_ACER

def evalute_threshold_based(prediction_scores, gt_labels, threshold):
    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    num_real = len([s for s in data if s['label'] == 1])
    num_fake = len([s for s in data if s['label'] == 0])

    type1 = len([s for s in data if s['map_score'] <= threshold and s['label'] == 1])
    type2 = len([s for s in data if s['map_score'] > threshold and s['label'] == 0])

    test_threshold_APCER = type2 / num_fake
    test_threshold_BPCER = type1 / num_real
    test_threshold_ACER = (test_threshold_APCER + test_threshold_BPCER) / 2.0

    return test_threshold_APCER, test_threshold_BPCER, test_threshold_ACER
