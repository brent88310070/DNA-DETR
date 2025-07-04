# ------------------------------------------------------------------------
# DNA-DETR: Copyright (c) 2025 SenseTime. All Rights Reserved.
# Licensed under the Apache License
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch

def box_cxsl_to_x1x2(x):
    cx, sl = x.unbind(-1)
    b = [(cx - 0.5 * sl), (cx + 0.5 * sl)]
    return torch.stack(b, dim=-1) #[bs, (x_start, x_end)]

def box_x1x2_to_cxsl(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (x1 - x0)]
    return torch.stack(b, dim=-1)

def box_center(x):
    x0, x1 = x.unbind(-1)
    b_c = [(x0 + x1) / 2]
    return torch.stack(b_c, dim=-1)


def diou_loss(boxes1, boxes2):
    # Expand dimensions for broadcasting
    boxes1_expanded = boxes1[:, None]
    boxes2_expanded = boxes2[None, :]

    i_s = torch.maximum(boxes1_expanded[:, :, :1], boxes2_expanded[:, :, :1])
    i_e = torch.minimum(boxes1_expanded[:, :, 1:], boxes2_expanded[:, :, 1:])
    intersect_d = torch.max(i_e - i_s, torch.zeros_like(i_e))

    box1_l = boxes1_expanded[:, :, 1:] - boxes1_expanded[:, :, :1]
    boxes2_l = boxes2_expanded[:, :, 1:] - boxes2_expanded[:, :, :1]
    union_d = box1_l + boxes2_l - intersect_d

    iou = intersect_d / union_d

    t_s = torch.minimum(boxes1_expanded[:, :, :1], boxes2_expanded[:, :, :1])
    t_e = torch.maximum(boxes1_expanded[:, :, 1:], boxes2_expanded[:, :, 1:])
    d2 = torch.pow(box_center(boxes1_expanded) - box_center(boxes2_expanded), 2)
    c2 = torch.pow(t_e - t_s, 2)

    a_diou_loss = 1 - iou + d2 / c2
    return a_diou_loss.squeeze()



def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
