import numpy as np
import sys
import scipy.spatial
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score

#evaluation functions
def metricEval(eval_metric, output, gt, num_labels):
    if eval_metric == 'iou':
        return get_iou(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'dice':
        return get_dice(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'recall':
        return get_recall(output.squeeze(), gt.squeeze(), num_labels)
    elif eval_metric == 'precision':
        return get_precision(output.squeeze(), gt.squeeze(), num_labels)
    else:
        print('Invalid evaluation metric value')
        sys.exit()
    print('MY IOU', get_iou(output.squeeze(), gt.squeeze(), num_labels))
    print('MY DICE', get_dice(output.squeeze(), gt.squeeze(), num_labels))
    print('MY recll', get_recall(output.squeeze(), gt.squeeze(), num_labels))
    print('MY PRECISION' , get_precision(output.squeeze(), gt.squeeze(), num_labels))
    print(precision_recall_fscore_support(gt.reshape(-1), output.reshape(-1)))

def get_iou(pred, gt, num_labels):
    if pred.shape != gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    max_label = num_labels-1
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        gt_loc = set(np.where(gt == j)[0])
        pred_loc = set(np.where(pred == j)[0])

        intersection = set.intersection(gt_loc, pred_loc)
        union = set.union(gt_loc, pred_loc)

        if len(gt_loc) != 0:
            count[j] = float(len(intersection)) / float(len(union))
    return np.sum(count) / float(num_labels)

def get_dice(pred, gt, num_labels):
    if num_labels != 2:
        print('Dice evaluation score is only implemented for 2 labels')
        sys.exit()
    return 1.0 - scipy.spatial.distance.dice(pred.reshape(-1), gt.reshape(-1))

#f1 score at beta = 1 is the same as dice score

# recall = (num detected WMH) / (num true WMH)
def get_recall(pred, gt, num_labels):
    if num_labels != 2:
        sys.exit()

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    gt_loc = set(np.where(gt == 1)[0])
    pred_loc = set(np.where(pred == 1)[0])
    TP = float(len(set.intersection(gt_loc, pred_loc)))
    TPandFN = float(len(gt_loc))
    return TP / TPandFN

# precision = (number detected WMH) / (number of all detections)
def get_precision(pred, gt, num_labels):
    if num_labels != 2:
        sys.exit()

    gt = gt.reshape(-1)
    pred = pred.reshape(-1)

    gt_loc = set(np.where(gt == 1)[0])
    pred_loc = set(np.where(pred == 1)[0])
    TP = float(len(set.intersection(gt_loc, pred_loc)))
    TPandFP = float(len(pred_loc))
    return TP / TPandFP