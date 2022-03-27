import cv2
import sklearn.metrics
#https://www.kite.com/blog/python/image-segmentation-tutorial/




def _assert_valid_lists(groundtruth_list, predicted_list):
    assert len(groundtruth_list) == len(predicted_list)
    for unique_element in np.unique(groundtruth_list).tolist():
        assert unique_element in [0, 1]


def _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [1]


def _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == np.unique(predicted_list).tolist() == [0]


def get_confusion_matrix_elements(groundtruth_list, predicted_list):
    """
    Return confusion matrix elements covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns confusion matrix elements i.e TN, FP, FN, TP in that order and as floats
    returned as floats to make it feasible for float division for further calculations on them
    """
    _assert_valid_lists(groundtruth_list, predicted_list)

    if _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = 0, 0, 0, np.float64(len(groundtruth_list))

    elif _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        tn, fp, fn, tp = np.float64(len(groundtruth_list)), 0, 0, 0

    else:
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(groundtruth_list, predicted_list).ravel()
        tn, fp, fn, tp = np.float64(tn), np.float64(fp), np.float64(fn), np.float64(tp)
    return tn, fp, fn, tp



def _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [0] and np.unique(predicted_list).tolist() == [1]


def _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return np.unique(groundtruth_list).tolist() == [1] and np.unique(predicted_list).tolist() == [0]


def _mcc_denominator_zero(tn, fp, fn, tp):
    _assert_valid_lists(groundtruth_list, predicted_list)
    return (tn == 0 and fn == 0) or (tn == 0 and fp == 0) or (tp == 0 and fp == 0) or (tp == 0 and fn == 0)


def get_f1_score(groundtruth_list, predicted_list):
    """
    Return f1 score covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns f1 score
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)

    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        f1_score = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        f1_score = 1
    else:
        f1_score = (2 * tp) / ((2 * tp) + fp + fn)

    return f1_score


def get_mcc(groundtruth_list, predicted_list):
    """
    Return mcc covering edge cases
    :param groundtruth_list list of groundtruth elements
    :param predicted_list list of predicted elements
    :return returns mcc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    if _all_class_0_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = 1
    elif _all_class_1_predicted_as_class_0(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _all_class_0_predicted_as_class_1(groundtruth_list, predicted_list) is True:
        mcc = -1
    elif _mcc_denominator_zero(tn, fp, fn, tp) is True:
        mcc = -1
    else:
        mcc = ((tp * tn) - (fp * fn)) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    mat_dict = {
        "tn" : tn,
        "fp" : fp,
        "fn" : fn,
        "tp" : tp
    }

    return mcc , mat_dict


def get_accuracy(groundtruth_list, predicted_list):
    """
    Return accuracy
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns accuracy
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    tn, fp, fn, tp = get_confusion_matrix_elements(groundtruth_list, predicted_list)
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    return accuracy


def get_validation_metrics(groundtruth_list, predicted_list):
    """
    Return validation metrics dictionary with accuracy, f1 score, mcc after
    comparing ground truth and predicted image
    :param groundtruth_list list of elements
    :param predicted_list list of elements
    :return returns a dictionary with accuracy, f1 score, and mcc as keys
    one could add other stats like FPR, FNR, TP, TN, FP, FN etc
    """
    _assert_valid_lists(groundtruth_list, predicted_list)
    validation_metrics = {}
    validation_metrics["accuracy"] = get_accuracy(groundtruth_list, predicted_list)
    validation_metrics["f1_score"] = get_f1_score(groundtruth_list, predicted_list)
    validation_metrics["mcc"]  , confusion_matrix = get_mcc(groundtruth_list, predicted_list)
    validation_metrics["confusion_matrix"] = confusion_matrix
    return validation_metrics


def get_confusion_matrix_intersection_mats(groundtruth, predicted):
    """
    Returns a dictionary of 4 boolean numpy arrays containing True at TP, FP, FN, TN.
    """
    confusion_matrix_arrs = {}
    groundtruth_inverse = np.logical_not(groundtruth)
    predicted_inverse = np.logical_not(predicted)
    confusion_matrix_arrs["tp"] = np.logical_and(groundtruth, predicted)
    confusion_matrix_arrs["tn"] = np.logical_and(groundtruth_inverse, predicted_inverse)
    confusion_matrix_arrs["fp"] = np.logical_and(groundtruth_inverse, predicted)
    confusion_matrix_arrs["fn"] = np.logical_and(groundtruth, predicted_inverse)
    return confusion_matrix_arrs


def get_confusion_matrix_overlaid_mask(image, groundtruth, predicted, alpha, colors):
    """
    Returns overlay the 'image' with a color mask where TP, FP, FN, TN are
    each a color given by the 'colors' dictionary
    """
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    masks = get_confusion_matrix_intersection_mats(groundtruth, predicted)
    color_mask = np.zeros_like(image)
    for label, mask in masks.items():
        color = colors[label]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    return cv2.addWeighted(image, alpha, color_mask, 1 - alpha, 0)




import os
import imageio
import numpy as np
gt_path = "/home/gaurav/Projects/medical/TeethU2Net/test_data_92/gt_masks"
mask_paths = "/home/gaurav/Projects/medical/TeethU2Net/Results/TeethNetMasks_results"


mat_dict = {
    "accuracy" : 0,
    "f1_score" : 0,
    "mcc" : 0 ,
    "tn" : 0,
    "fp" : 0,
    "fn" : 0,
    "tp" : 0
}
total_data = len(os.listdir(gt_path))
for image_name in os.listdir(gt_path):
    gt_image_path = os.path.join(gt_path , image_name)
    mask_path = os.path.join(mask_paths , image_name.replace('.bmp' ,'.png'))
    print(mask_path)
    gt_image =imageio.imread(gt_image_path)
    mask_image = imageio.imread(mask_path , as_gray=True)
    groundtruth_scaled = gt_image // 255
    predicted_scaled = mask_image // 255

    groundtruth_list = (groundtruth_scaled).flatten().tolist()
    predicted_list = (predicted_scaled).flatten().tolist()
    validation_metrics = get_validation_metrics(groundtruth_list, predicted_list)
    mat_dict["accuracy"] += validation_metrics['accuracy']
    mat_dict['f1_score'] += validation_metrics['f1_score']
    mat_dict['mcc'] += validation_metrics['mcc']

    mat_dict['tn'] += validation_metrics['confusion_matrix']['tn']
    mat_dict['fp'] += validation_metrics['confusion_matrix']['fp']
    mat_dict['fn'] += validation_metrics['confusion_matrix']['fn']
    mat_dict['tp'] += validation_metrics['confusion_matrix']['tp']


    print("Validation Metrics comparing Otsu and ground truth")
    print(validation_metrics)

results = dict()
for key, value in mat_dict.items():
    results[key] = value/total_data

print("Final Matrix :",results)