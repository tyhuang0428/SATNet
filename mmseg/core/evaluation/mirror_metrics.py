import numpy as np
import os
import math
from tqdm import tqdm


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_eval(results, gt_seg_maps):
    ACC = []
    IOU = []
    precision_record, recall_record = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
    MAE = []
    BER = []

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    for i in tqdm(range(num_imgs)):
        pred = results[i]
        gt = gt_seg_maps[i]
        acc = accuracy_mirror(pred, gt)
        iou = compute_iou(pred, gt)
        precision, recall = cal_precision_recall((pred * 255).astype(np.uint8), (gt * 255).astype(np.uint8))
        for idx, data in enumerate(zip(precision, recall)):
            p, r = data
            precision_record[idx].update(p)
            recall_record[idx].update(r)
        mae = compute_mae(pred, gt)
        ber = compute_ber(pred, gt)

        ACC.append(acc)
        IOU.append(iou)
        if math.isnan(mae) is False:
            MAE.append(mae)
        BER.append(ber)

    mean_ACC = sum(ACC) / len(ACC)
    mean_IOU = 100 * sum(IOU) / len(IOU)
    F = cal_fmeasure([precord.avg for precord in precision_record], [rrecord.avg for rrecord in recall_record])
    mean_MAE = sum(MAE) / len(MAE)
    mean_BER = 100 * sum(BER) / len(BER)
    return mean_ACC, mean_IOU, F, mean_MAE, mean_BER


def accuracy_mirror(predict_mask, gt_mask):
    """
    sum_i(n_ii) / sum_i(t_i)
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = TP/N_p

    return accuracy_


def compute_iou(predict_mask, gt_mask):
    """
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    Here, n_cl = 1 as we have only one class (mirror).
    :param predict_mask:
    :param gt_mask:
    :return:
    """

    check_size(predict_mask, gt_mask)

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        IoU = 0

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_


def cal_precision_recall(predict_mask, gt_mask):
    # input should be np array with data type uint8
    assert predict_mask.dtype == np.uint8
    assert gt_mask.dtype == np.uint8
    assert predict_mask.shape == gt_mask.shape

    eps = 1e-4

    prediction = predict_mask / 255.
    gt = gt_mask / 255.

    # mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt).astype(np.float32)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt).astype(np.float32)
        p = np.sum(hard_prediction).astype(np.float32)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


def compute_mae(predict_mask, gt_mask):

    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    mae_ = np.mean(abs(predict_mask - gt_mask)).item()

    return mae_


def compute_ber(predict_mask, gt_mask):
    """
    BER: balance error rate.
    :param predict_mask:
    :param gt_mask:
    :return:
    """
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    # if N_p + N_n != 640 * 512:
    #     raise Exception("Check if mask shape is correct!")

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    ber_ = 1 - (1 / 2) * ((TP / N_p) + (TN / N_n))

    return ber_


def segm_size(segm):
    try:
        height = segm.shape[0]
        width  = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''
class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
