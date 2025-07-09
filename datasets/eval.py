# ------------------------------------------------------------------------
# DNA-DETR: Copyright (c) 2025 SenseTime. All Rights Reserved.
# Licensed under the Apache License
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
evaluator
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from collections import Counter

def calculate_iou(pred_bbox, target_bbox):
    #(center, len) to (bbox_start, bbox_end)
    pred_min, pred_max = pred_bbox[0] - pred_bbox[1]/2, pred_bbox[0] + pred_bbox[1]/2
    target_min, target_max = target_bbox[0] - target_bbox[1]/2, target_bbox[0] + target_bbox[1]/2

    intersection_start = max(pred_min, target_min)
    intersection_end = min(pred_max, target_max)

    intersection_length = max(0, intersection_end - intersection_start)
    union_length = (pred_max - pred_min) + (target_max - target_min) - intersection_length

    iou = intersection_length / union_length
    return iou

def calculate_max_iou(pred_bbox, target_bboxes):
    max_iou = 0
    for i in range(0, len(target_bboxes)):
        iou = calculate_iou(pred_bbox, target_bboxes[i])
        if max_iou <= iou:
            max_iou = iou
    return max_iou


def eval_record(outputs, targets, eval_result_dict):
    nor_prob = F.softmax(outputs['pred_logits'], -1) #normalization to probability

    for seq in range(0, len(targets)):

        # For FN
        labels = list(set([int(tensor.item()) for tensor in targets[seq]["labels"]]))
        for label in labels:
            if label == 0:
                break
            class_list = torch.argmax(nor_prob[seq], dim=1) == label
            DNA_indices = torch.nonzero(class_list).squeeze().tolist()
            if isinstance(DNA_indices, int):
                DNA_indices = [DNA_indices]
            if len(DNA_indices) == 0:
                eval_result_dict[label]["num_FN"] += 1   


        for tar in range(0, len(targets[seq]["labels"])):
            DNA_type = targets[seq]["labels"][tar].item()
            class_list = torch.argmax(nor_prob[seq], dim=1) == DNA_type #Choose the higher probabilities
            DNA_indices = [i for i, val in enumerate(class_list) if val] #Get the DNA idx

            for idx in DNA_indices:
                pred_bbox = outputs['pred_boxes'][seq][idx]
                target_bboxes = targets[seq]["boxes"]
                max_iou = calculate_max_iou(pred_bbox, target_bboxes)

                prob = nor_prob[seq][idx][DNA_type]
                conf = prob * max_iou

                eval_result_dict[DNA_type]["IoU"].append(max_iou.item())
                eval_result_dict[DNA_type]["Prob"].append(prob.item())
                eval_result_dict[DNA_type]["Confidence"].append(conf.item())

    return eval_result_dict


def record_confusion(true_value, predicted_value):
    if true_value == True and predicted_value == True:
        return 1 #TP
    elif true_value == False and predicted_value == False:
        return 2 #TN
    elif true_value == False and predicted_value == True:
        return 3 #FP
    elif true_value == True and predicted_value == False:
        return 4 #FN


def class_record(outputs, targets, class_result_dict, data_num_class):
    for seq in range(0, len(targets)):
        logits_tensor = torch.argmax(outputs['pred_logits'][seq], dim=1)
        pos_num = torch.sum((logits_tensor != 0) & (logits_tensor != data_num_class)) #data_num_class for no object
        # TP
        if pos_num > 0 and targets[seq]["labels"][0] != 0:
            class_result_dict['TP'] += 1
        # FP
        if pos_num == 0 and targets[seq]["labels"][0] != 0:
            class_result_dict['FP'] += 1
        # FN
        if pos_num > 0 and targets[seq]["labels"][0] == 0:
            class_result_dict['FN'] += 1
        # TN
        if pos_num == 0 and targets[seq]["labels"][0] == 0:
            class_result_dict['TN'] += 1

    return class_result_dict



def calculate_ap(eval_result_dict):
    ap_dict = {i: [] for i in range(0, len(eval_result_dict))}
    precision_dict = {i: [1.0] for i in range(0, len(eval_result_dict))}
    recall_dict = {i: [0] for i in range(0, len(eval_result_dict))}
    for DNA_type in range(0, len(eval_result_dict)):
        # Sort predictions by confidence in descending order
        confidence_list = eval_result_dict[DNA_type]['Confidence']
        sorted_indices = sorted(range(len(confidence_list)), key=lambda k: confidence_list[k], reverse=True)
        iou_list = eval_result_dict[DNA_type]['IoU']
        iou_list = [iou_list[i] for i in sorted_indices]

        iou_threshold = 0.5 # IoU > 0.5 as True Positive
        true_positives, false_positives = 0, 0 

        total_true_objects = len(iou_list) + eval_result_dict[DNA_type]["num_FN"]
        # print(DNA_type, " num_FN: ", eval_result_dict[DNA_type]["num_FN"])
        # print(DNA_type, " iou_list: ", len(iou_list))

        for i in range(len(iou_list)):
            if iou_list[i] > iou_threshold:
                true_positives += 1
            else:
                false_positives += 1

            precision_dict[DNA_type].append(true_positives / (true_positives + false_positives))
            recall_dict[DNA_type].append(true_positives / total_true_objects)

        # Calculate AP using trapezoidal rule
        ap = 0.0
        for i in range(1, len(precision_dict[DNA_type])):
            ap += (recall_dict[DNA_type][i] - recall_dict[DNA_type][i - 1]) * precision_dict[DNA_type][i]
        ap_dict[DNA_type] = ap

    return ap_dict


def calculate_class_performance(class_result):
    confusion_matrix = Counter(class_result)
    TP = confusion_matrix[1]
    TN = confusion_matrix[2]
    FP = confusion_matrix[3]
    FN = confusion_matrix[4]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1
