# ------------------------------------------------------------------------
# DNA-DETR: Copyright (c) 2025 SenseTime. All Rights Reserved.
# Licensed under the Apache License
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=30, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_diou', default=2, type=float,
                        help="diou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--diou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.05, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--feature_type', default='both', help="one-hot | dot | both")
    parser.add_argument('--dataset_file', default='nonb')
    parser.add_argument('--data_num_class', default=6, type=int,
                        help="for a dataset that has a single class with id 1, you should pass `num_classes` to be 2 (max_obj_id + 1)")
    parser.add_argument('--data_path', default='./data/both/', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    model.to(device)
    print(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(seq_set='train', args=args)
    print('Build train dataset already!')
    dataset_val = build_dataset(seq_set='val', args=args)
    print('Build validation dataset already!')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, shuffle=True)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        ap_list = []
        test_stats, ap, class_result = evaluate(model, criterion, data_loader_val, device, args.output_dir, args)

        performance_df = pd.DataFrame(columns=['AP'])
        entry = {'AP': ap}
        performance_df = pd.concat([performance_df, pd.DataFrame([entry])], ignore_index=True)
        performance_df.to_csv('performance_metrics.csv', index=False)

        print("Class results: ", class_result)
        print(ap)
        ap_list.append(ap)
        return ap_list

    # Loss write to TensorBoard
    writer = SummaryWriter(Path(args.output_dir, "runs").as_posix())

    print("Start training")
    start_time = time.time()
    ap_list = []
    mAP_list = []
    performance_df = pd.DataFrame(columns=['Epoch', 'AP'])
    MAX_mAP = 0
    BEST_EPOCH = -1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, writer = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, writer, 
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, ap, class_result = evaluate(
            model, criterion, data_loader_val, device, args.output_dir, args
        )

        mAP = sum(ap.values()) / (args.data_num_class - 1)
        mAP_list.append(mAP)

        if args.output_dir and mAP >= MAX_mAP:
            max_ap_checkpoint_path = [output_dir / 'max_ap.pth']
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, max_ap_checkpoint_path[0])
            MAX_mAP = mAP

        print("Epoch", epoch, " mAP:", mAP, " AP:", ap)
        ap_list.append(ap)
        print("Class results: ", class_result)

        entry = {'Epoch': epoch, 'AP': ap}
        performance_df = pd.concat([performance_df, pd.DataFrame([entry])], ignore_index=True)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f'Training time: {datetime.timedelta(seconds=int(total_time))}')
    print(f'Best mAP {MAX_mAP:.4f} @ epoch {BEST_EPOCH}')
    performance_df.to_csv(output_dir / 'performance_metrics.csv', index=False)
    writer.close()
    return ap_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ap_list = main(args)

    max_mAP = 0
    for i in range(len(ap_list)):
        mAP = sum(ap_list[i].values()) / (args.data_num_class - 1)
        print("Epoch ", i, " : ", mAP)
        if max_mAP <= mAP:
            max_mAP = mAP
            max_mAP_epoch = i
    print("Max AP Epoch ", max_mAP_epoch, " : ", max_mAP)  
    print("Finish!")
