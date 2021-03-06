# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
import datetime
import time
import math
import json
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import learn2learn as l2l
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
# from dataset.customData import customData
from dataset.customJsonData import customData, get_inf_iterator

import utils
import random
import vision_transformer as vits
from vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():    
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'deit_tiny', 'deit_small'] + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--pretrained_weights', default='./output/ssl_celebA/checkpoint.pth', type=str, help="Path to pretrained weights to finetune.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--gpu', default=[0,1])
    parser.add_argument('--batch_size_per_gpu', default=18, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=0, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=3, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='../../CelebA_Data/trainSquareCropped', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--txt_path", default='../../CelebA_Data/metas/intra_test/train_label.txt', type=str)
    parser.add_argument("--json_path", default='../../CelebA_Data/metas/intra_test/train_label.json', type=str)
    parser.add_argument('--output_dir', default="output/ssl_meta_preTrain_DINO_gpu_2_bs_45_lr_ori/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")    
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    #dataset = datasets.ImageFolder(args.data_path, transform=transform)
    # ????????????????????? 5 ??? Task????????????????????? Task ????????????????????????????????????????????? Task??????????????????????????????????????? Task ???????????????????????? Dataset
    # ??????????????????????????? Label ???????????? SSL!? ??????
    # ???????????????????????????????????? Linear classifier ????????????
    dataset_live = customData(illumination_domain=0,
                                img_path=args.data_path,
                                txt_path=args.txt_path,
                                json_path=args.json_path,
                                data_transforms=transform,
                                phase="ssl")

    dataset_normal = customData(illumination_domain=1,
                                img_path=args.data_path,
                                txt_path=args.txt_path,
                                json_path=args.json_path,
                                data_transforms=transform,
                                phase="ssl")

    dataset_strong = customData(illumination_domain=2,
                                img_path=args.data_path,
                                txt_path=args.txt_path,
                                json_path=args.json_path,
                                data_transforms=transform,
                                phase="ssl")        

    dataset_back = customData(illumination_domain=3,
                                img_path=args.data_path,
                                txt_path=args.txt_path,
                                json_path=args.json_path,
                                data_transforms=transform,
                                phase="ssl")                                                    

    dataset_dark = customData(illumination_domain=4,
                                img_path=args.data_path,
                                txt_path=args.txt_path,
                                json_path=args.json_path,
                                data_transforms=transform,
                                phase="ssl")

    sampler_live = torch.utils.data.DistributedSampler(dataset_live, shuffle=True)
    data_loader_live = torch.utils.data.DataLoader(
        dataset_live,
        sampler=sampler_live,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    sampler_normal = torch.utils.data.DistributedSampler(dataset_normal, shuffle=True)
    data_loader_normal = torch.utils.data.DataLoader(
        dataset_normal,
        sampler=sampler_normal,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    sampler_strong = torch.utils.data.DistributedSampler(dataset_strong, shuffle=True)
    data_loader_strong = torch.utils.data.DataLoader(
        dataset_strong,
        sampler=sampler_strong,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    sampler_back = torch.utils.data.DistributedSampler(dataset_back, shuffle=True)
    data_loader_back = torch.utils.data.DataLoader(
        dataset_back,
        sampler=sampler_back,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    sampler_dark = torch.utils.data.DistributedSampler(dataset_dark, shuffle=True)
    data_loader_dark = torch.utils.data.DataLoader(
        dataset_dark,
        sampler=sampler_dark,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    iternum = max(len(data_loader_live),len(data_loader_normal),
                  len(data_loader_strong),len(data_loader_back), 
                  len(data_loader_dark))    
     
    print(f"Data loaded: there are {len(dataset_live)} live images.")
    print(f"Data loaded: there are {len(dataset_normal)} normal images.")
    print(f"Data loaded: there are {len(dataset_strong)} strong images.")
    print(f"Data loaded: there are {len(dataset_back)} back images.")
    print(f"Data loaded: there are {len(dataset_dark)} dark images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=0.1,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # load pretrain on oringinal DINO model
    utils.load_pretrained_weights(student, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    utils.load_pretrained_weights(teacher, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else: # default
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, iternum,
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, iternum,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, iternum)
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]
    writer = SummaryWriter()
    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader_live.sampler.set_epoch(epoch)
        data_loader_normal.sampler.set_epoch(epoch)
        data_loader_strong.sampler.set_epoch(epoch)
        data_loader_back.sampler.set_epoch(epoch)
        data_loader_dark.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, embed_dim, dino_loss,
            iternum, data_loader_live, data_loader_normal, data_loader_strong, data_loader_back, data_loader_dark,
            optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        for k,v in train_stats.items():
            writer.add_scalar(str(k), v, epoch)
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, embed_dim, dino_loss, 
                    iternum, data_loader_live, data_loader_normal, data_loader_strong, data_loader_back, data_loader_dark,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
      
    data_live = get_inf_iterator(data_loader_live)
    data_normal = get_inf_iterator(data_loader_normal)
    data_strong = get_inf_iterator(data_loader_strong)
    data_back = get_inf_iterator(data_loader_back)
    data_dark = get_inf_iterator(data_loader_dark)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    adapt_opt_state = optimizer.state_dict()# ???????????????????????? update student
    for step in range(iternum):
        iterLoss = 0
        optimizer.zero_grad()
        # zero-grad the parameters
        for p in student.parameters():
            p.grad = torch.zeros_like(p.data)

        metric_logger.log_every_meta(step, iternum, 10, header)

        # ============ preparing meta-train???meta-test data ... ============
        live_images, _ = next(data_live)
        normal_images, _ = next(data_normal)
        strong_images, _ = next(data_strong)
        back_images, _ = next(data_back)
        dark_images, _ = next(data_dark)

        it = iternum * epoch + step
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        catimglist = [live_images, normal_images, strong_images, back_images, dark_images]
        domain_list = list(range(len(catimglist)))
        random.shuffle(domain_list)
        meta_train_list = domain_list[:len(catimglist)-1] 
        meta_test_list = domain_list[len(catimglist)-1:]

        # ?????? meta-test image ??????????????????????????????
        meta_test_index = meta_test_list[0]
        meta_test_images = catimglist[meta_test_index]
        # move images to gpu
        meta_test_images = [im.cuda(non_blocking=True) for im in meta_test_images]
        
        for index in meta_train_list: # iterate all tasks, each task include meta-train and meta-test
            
            # ============ building student learner networks ... ============
            '''
            student_vit_learner = copy.deepcopy(student.module.backbone)# ???????????????????????? copy ????????????????????? student_grad?????????????????? adapt_opt.param_groups ??????
            student_learner = utils.MultiCropWrapper(
                student_vit_learner, 
                DINOHead(
                embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            ))
            student_learner = student_learner.cuda()
            nn.parallel.DistributedDataParallel(student_learner, device_ids=[args.gpu])
            student_learner.head.mlp = copy.deepcopy(student.module.head.mlp)
            # w = g * v / |v|??????????????? g ??? 1 
            student_learner.head.last_layer.weight.copy_(student.module.head.last_layer.weight) #non-leaf, no grad
            student_learner.head.last_layer.weight_g.copy_(student.module.head.last_layer.weight_g) #leaf, no grad
            student_learner.head.last_layer.weight_v = copy.deepcopy(student.module.head.last_layer.weight_v) #leaf, has grad            
            '''
            student_learner = student

            # ============ building teacher learner networks ... ============
            '''
            teacher_vit_learner = copy.deepcopy(teacher.backbone)
            teacher_learner = utils.MultiCropWrapper(
                teacher_vit_learner,
                DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
            )
            teacher_learner = teacher_learner.cuda()
            teacher_learner.head.mlp = copy.deepcopy(teacher.head.mlp)
            teacher_learner.head.last_layer.weight.copy_(teacher.head.last_layer.weight) #non-leaf, no grad
            teacher_learner.head.last_layer.weight_g.copy_(teacher.head.last_layer.weight_g) #leaf, no grad
            teacher_learner.head.last_layer.weight_v = copy.deepcopy(teacher.head.last_layer.weight_v) #leaf, has grad
            '''
            teacher_learner = teacher
            # print(student.module.backbone)
            # print(student.module.head.mlp[4])
            # print(student.module.head.mlp[4].weight.grad_fn)# None, ?????????????????????
            # print(student.module.head.last_layer.weight_g.grad_fn)# None, ??????????????????????????????????????? 1???????????? gradient_func            
            
            # ============ preparing optimizer ... ============
            params_groups = utils.get_params_groups(student_learner)
            if args.optimizer == "adamw":
                adapt_opt = torch.optim.AdamW(params_groups)  # to use with ViTs
            elif args.optimizer == "sgd":
                adapt_opt = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
            elif args.optimizer == "lars":
                adapt_opt = utils.LARS(params_groups)  # to use with convnet and large batches
            
            # adapt_opt.load_state_dict(adapt_opt_state) # ???????????? load???????????? load ?????????????????? param_group ?????? learner ???
            # adapt_opt.param_groups = utils.get_params_groups(student_learner)# error, amsgrad ?????????
            # ????????????????????? lr???w_decay???????????????????????????
            for i, param_group in enumerate(adapt_opt.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # catimg_meta = catimglist[index]
            # batchidx = list(range(len(catimg_meta)))
            # random.shuffle(batchidx)
            # images = catimg_meta[batchidx,:]
            # move images to gpu
            """
            Meta-train
            """
            images = catimglist[index]
            images = [im.cuda(non_blocking=True) for im in images]
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher_learner(images[:2])  # only the 2 global views pass through the teacher
                student_output = student_learner(images)
                adapt_opt.zero_grad()
                loss = dino_loss(student_output, teacher_output, epoch)
                iterLoss += loss

            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad: #to prevent diverged training
                    param_norms = utils.clip_gradients(student_learner, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student_learner,
                                                args.freeze_last_layer)
                adapt_opt.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(adapt_opt)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student_learner, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student_learner,
                                                args.freeze_last_layer) # ???????????????????????? last_layer weight_v ??? gradient
                fp16_scaler.step(adapt_opt)
                fp16_scaler.update()
            # loss.backward()
            # adapt_opt.step()
            with torch.no_grad(): # EMA ?????? teacher_learner
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student_learner.parameters(), teacher_learner.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            # for p, l in zip(student.parameters(), student_learner.parameters()): # ???????????????????????????????????? meta-test ??????????????????????????????
            #     p.grad.data.add_(-1.0, l.data)
            """
            Meta-test
            """            
            # catimg_meta = catimglist[meta_test_index]
            # batchidx = list(range(len(catimg_meta)))
            # random.shuffle(batchidx)
            # images = catimg_meta[batchidx,:]
            
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                teacher_output = teacher_learner(meta_test_images[:2])  # only the 2 global views pass through the teacher
                student_output = student_learner(meta_test_images)
                adapt_opt.zero_grad()
                loss = dino_loss(student_output, teacher_output, epoch)
                iterLoss += loss

            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad: #to prevent diverged training
                    param_norms = utils.clip_gradients(student_learner, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student_learner,
                                                args.freeze_last_layer)
                adapt_opt.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad:
                    fp16_scaler.unscale_(adapt_opt)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student_learner, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student_learner,
                                                args.freeze_last_layer)
                fp16_scaler.step(adapt_opt)
                fp16_scaler.update()
            # loss.backward()
            # adapt_opt.step() # ?????? student_learner?????????????????? student ?????? student_learner ??????
            # ????????????????????? teacher_learner ???????????????????????? meta-test

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
            for p, l in zip(student.parameters(), student_learner.parameters()): # ????????????????????? get_params_groups
                p.grad.data.add_(-1.0, l.data)# student.grad = (-1)*student_LR_W?????????????????? 4 ???

        # ============ finish meta-train and meta-test. student, teacher update ... ============
        # optimizer.zero_grad() # ???????????? zero_grad ?????? student.grad ????????????!?
        updateTimes = len(catimglist)-1
        for p in student.parameters():
            # tmpA = p.grad.data
            # tmpB = p.grad.data.mul_(1.0 / float(updateTimes))# ??????????????? 5 ???????????? (ex:BC????????????-0.000025)
            # tmpC = p.data
            p.grad.data.mul_(1.0 / updateTimes).add_(p.data) # student_W = [4*(-1)*student_LR_W]/4 + student_W
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=iterLoss.item())# loss ???????????? iterLoss??????????????????????????? iteration total loss
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
