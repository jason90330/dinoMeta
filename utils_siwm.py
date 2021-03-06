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
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
import os
import sys
import time
import math
import random
import datetime
import subprocess
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from sklearn.metrics import roc_curve, auc, confusion_matrix


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=True)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log_every_meta(self, iter_now, iterable_len, print_freq, header=None):
        # i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(iterable_len))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        # for iterIdx in range(0,iterable_len):
        data_time.update(time.time() - end)
        iter_time.update(time.time() - end)
        if iter_now % print_freq == 0 or iter_now == iterable_len - 1:
            eta_seconds = iter_time.global_avg * (iterable_len - iter_now)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if torch.cuda.is_available():
                print(log_msg.format(
                    iter_now, iterable_len, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            else:
                print(log_msg.format(
                    iter_now, iterable_len, eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
        # i += 1
        # end = time.time()
        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('{} Total time: {} ({:.6f} s / it)'.format(
        #     header, total_time_str, total_time / iterable_len))

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    

def model_eval(output, target, output_dir, epoch):
    output_txt_dir = output_dir+"/txt/"
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    acer_avg = 0
    # best_acer_avg = 1
    target = target.cpu().detach().numpy().astype(int).copy()
    probablity = torch.nn.functional.softmax(output, dim=1).cpu().detach().numpy().copy()
    # probablity = probablity[:,1:]
    # category_out = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # category_tar = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # for out, tar in enumerate(zip(output, target)):
    #     category_out[tar].append(out)
    #     category_t[tar].append(out)
    with open(output_txt_dir + "score.txt","a+") as f:
        f.write("="*60)
        f.write('\nModel %03d \n'%(epoch))
        gt_labels = np.zeros(shape=output.shape,dtype='int')
        pd_labels = np.zeros(shape=output.shape,dtype='int')
        for idx, (tar,probs) in enumerate(zip(target,probablity)):
            gt_labels[idx][tar] = 1
            predict_label = np.argmax(probs)
            pd_labels[idx][predict_label] = 1
        cm = confusion_matrix(gt_labels.argmax(axis=1), pd_labels.argmax(axis=1))
        plt.figure(figsize=(16, 12))
        labelName = ['Live', 
                    'Replay', 'Print', 
                    '3D_M_Half', '3D_M_Silic.', '3D_M_Trans.', '3D_M_Paper', '3D_M_Manne.', 
                    'Mkup_Ob.', 'Mkup_lm.', 'Mkup_Cos.', 
                    'Par_Att_Fun.', 'Par_Att_Papergls.', 'Par_Att_Paper']
        plot_confusion_matrix(cm, classes=labelName, normalize=True,
                        title="confusion matrix")
        # plt.savefig(""cfg.CONFUSION_PATH)
        plt.savefig("%s/CM_norm%03d.png" %(output_dir, epoch))
        
        plt.figure(figsize=(16, 12))
        plot_confusion_matrix(cm, classes=labelName, normalize=False,
                        title="confusion matrix")
        plt.savefig("%s/CM_%03d.png" %(output_dir, epoch))
        for spoof_idx in range(1, 14):
            fdr, tdr, _ = roc_curve(gt_labels[:, spoof_idx], probablity[:, spoof_idx])
            tmp = gt_labels[:, spoof_idx]
            roc_auc = auc(fdr, tdr)
            fnr = 1-tdr
            diff = np.absolute(fnr - fdr)
            idx = np.nanargmin(diff)
            # print(threshold[idx])
            eer = np.mean((fdr[idx],fnr[idx]))        

            avg = np.add(fdr, fnr)
            idx = np.nanargmin(avg)
            hter = np.mean((fdr[idx],fnr[idx])) 

            # fpr_at_10e_m3_idx = np.argmin(np.abs(fpr-10e-3))
            # tpr_cor_10e_m3 = tpr[fpr_at_10e_m3_idx+1]

            fpr_at_5e_m3_idx = np.argmin(np.abs(fdr-5e-3))
            while fdr[fpr_at_5e_m3_idx+1]==0:
                fpr_at_5e_m3_idx+=1
            tpr_cor_5e_m3 = tdr[fpr_at_5e_m3_idx+1]
            if tpr_cor_5e_m3 < 0.1 and spoof_idx == 6:
                fdr-5e-3
                print(fpr_at_5e_m3_idx)
                print("FDR:")
                for i in fdr[:10]:
                    print(i)
                print("TDR")
                for j in tdr[:10]:
                    print(tdr)
                print("*"*60)

            # fpr_at_10e_m4_idx = np.argmin(np.abs(fpr-10e-4))
            # tpr_cor_10e_m4 = tpr[fpr_at_10e_m4_idx+1]

            # actual = list(map(lambda el:[el], category_target))
            # category_pred = list(map(lambda el:[el], category_pred))
            
            
            TP = cm[spoof_idx][spoof_idx]
            FP = np.sum(cm, axis=0)[spoof_idx] - TP
            FN = np.sum(cm, axis=1)[spoof_idx] - TP
            TN = np.sum(cm) - FP - FN - TP
            tmp = np.sum(cm)
            accuracy = ((TP+TN))/(TP+FN+FP+TN)
            recall = (TP)/(TP+FN)
            apcer = FP/(TN+FP)
            bpcer = FN/(FN+TP)
            acer = (apcer+bpcer)/2
            acer_avg += acer
            f.write('\nSpoof type: %03d \n'%(spoof_idx))
            f.write('TP:%d, TN:%d,  FP:%d,  FN:%d\n' %(TP,TN,FP,FN))
            f.write('accuracy:%f\n'%(accuracy))
            f.write('recall:%f\n'%(recall))
            f.write('apcer:%f\n'%(apcer))
            f.write('bpcer:%f\n'%(bpcer))
            f.write('acer:%f\n'%(acer))
            f.write('eer:%f\n'%(eer))
            f.write('hter:%f\n'%(hter))
            f.write('TPR@FPR=5E-3:%f\n'%(tpr_cor_5e_m3))

        acer_avg/=13
        f.write('acer average:%f\n'%(acer_avg))
        # if acer_avg < best_acer_avg:
        #     with open(output_dir + "best_score.txt","w") as f_best:
        #         f_best.write("EPOCH=%d,best acer average= %.3f%%" % (epoch + 1, acer_avg))
        #         f_best.close()
        #         best_acer_avg = acer_avg
        '''
        for eval_spoof_type in range(1, 14):            
            category_score = []
            category_target = []
            category_pred = []
            f.write('\nSpoof type: %03d \n'%(eval_spoof_type))
            for idx,(out, tar) in enumerate(zip(probablity, target)):
                pred = np.argmax(out)
                if tar == 0:
                    category_target.append(0)
                    category_score.append(out[0]) #?????? live index ????????? probability
                    if pred == tar: # ????????????????????????
                        category_pred.append(0)#????????? live
                    else:
                        category_pred.append(1)#????????????????????? spoof

                elif tar == eval_spoof_type:
                    category_target.append(1)
                    category_score.append(out[tar]) #???????????? spoof target index ????????? probability
                    if pred == tar: # ????????????????????????
                        category_pred.append(1)#????????? spoof
                    else:
                        category_pred.append(0)#???????????????????????? live(?????????????????????????????????????????????type???spoof)

            # score = probablity[:,1:]#np.squeeze(score, 1)
            # pred = np.round(category_score)
            # category_pred = np.squeeze(category_pred, 0)
            category_pred = np.array(category_pred, dtype=int)
            # _, pred = output.topk(1, 1, True, True)
            # pred = pred.t()            
                # calculate eer
                
            # fpr, tpr, threshold = roc_curve(category_target,category_score)          
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            fnr = 1-tpr
            diff = np.absolute(fnr - fpr)
            idx = np.nanargmin(diff)
            # print(threshold[idx])
            eer = np.mean((fpr[idx],fnr[idx]))        

            avg = np.add(fpr, fnr)
            idx = np.nanargmin(avg)
            hter = np.mean((fpr[idx],fnr[idx])) 

            # fpr_at_10e_m3_idx = np.argmin(np.abs(fpr-10e-3))
            # tpr_cor_10e_m3 = tpr[fpr_at_10e_m3_idx+1]

            fpr_at_5e_m3_idx = np.argmin(np.abs(fpr-5e-3))
            tpr_cor_5e_m3 = tpr[fpr_at_5e_m3_idx+1]

            # fpr_at_10e_m4_idx = np.argmin(np.abs(fpr-10e-4))
            # tpr_cor_10e_m4 = tpr[fpr_at_10e_m4_idx+1]

            actual = list(map(lambda el:[el], category_target))
            category_pred = list(map(lambda el:[el], category_pred))
            
            cm = confusion_matrix(actual, category_pred)
            TP = cm[0][0]
            TN = cm[1][1]
            FP = cm[1][0]
            FN = cm[0][1]
            accuracy = ((TP+TN))/(TP+FN+FP+TN)
            recall = (TP)/(TP+FN)
            apcer = FP/(TN+FP)
            bpcer = FN/(FN+TP)
            acer = (apcer+bpcer)/2
            f.write('TP:%d, TN:%d,  FP:%d,  FN:%d\n' %(TP,TN,FP,FN))
            f.write('accuracy:%f\n'%(accuracy))
            f.write('recall:%f\n'%(recall))
            f.write('apcer:%f\n'%(apcer))
            f.write('bpcer:%f\n'%(bpcer))
            f.write('acer:%f\n'%(acer))
            f.write('eer:%f\n'%(eer))
            f.write('hter:%f\n'%(hter))
            f.write('TPR@FPR=5E-3:%f\n'%(tpr_cor_5e_m3))
        '''

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False
