from __future__ import print_function
from sklearn.datasets import fetch_mldata
import logging
import copy
from datetime import datetime

import argparse
import cPickle
import os
import os.path as osp
import re
import sys
import math
import time
from functools import partial
from PIL import Image
from multiprocessing import Pool

import numpy as np

import mxnet as mx
import scipy.io

from util import mxutil
from util import transformer as ts
from util import util
from util.lr_scheduler import FixedScheduler, LinearScheduler, PolyScheduler

from data_mix_PatchMine import FileIter, make_divisible

def parse_split_file_tgt(dataset_tgt, split_tgt, data_root=''):
    split_filename = 'issegm/data/{}/{}.lst'.format(dataset_tgt, split_tgt)
    image_list = []
    label_gt_list = []
    image_data_list = []
    
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
            image_data_list.append(fields[0])
            label_gt_list.append(os.path.join(data_root, fields[1]))
            
    return image_list, label_gt_list,image_data_list

def parse_model_label(args):
    assert args.model is not None
    fields = [_.strip() for _ in osp.basename(args.model).split('_')]
    # parse fields
    i = 0
    num_fields = len(fields)
    # database
    dataset = fields[i] if args.dataset is None else args.dataset
    dataset_tgt = args.dataset_tgt
    i += 1
    ######################## network structure
    assert fields[i].startswith('rn')
    net_type = re.compile('rn[a-z]*').findall(fields[i])[0]
    net_name = fields[i][len(net_type):].strip('-')
    i += 1
    # number of classes
    assert fields[i].startswith('cls')
    classes = int(fields[i][len('cls'):])
    i += 1
    ######################## feature resolution
    feat_stride = 32
    if i < num_fields and fields[i].startswith('s'):
        feat_stride = int(fields[i][len('s'):])
        i += 1

    # learning rate
    lr_params = {
        'type': 'fixed',
        'base': 0.1,
        'args': None,
    }
    if args.base_lr is not None:
        lr_params['base'] = args.base_lr
    if args.lr_type in ('linear',):
        lr_params['type'] = args.lr_type
    elif args.lr_type in ('poly',):
        lr_params['type'] = args.lr_type
    elif args.lr_type == 'step':
        lr_params['args'] = {'step': [int(_) for _ in args.lr_steps.split(',')],
                             'factor': 0.1}

    model_specs = {
        # model
        'lr_params': lr_params,
        'net_type': net_type,
        'net_name': net_name,
        'classes': classes,
        'feat_stride': feat_stride,
        # data
        'dataset': dataset,
        'dataset_tgt': dataset_tgt
    }
    return model_specs


def parse_args():
    parser = argparse.ArgumentParser(description='Tune FCRNs from ResNets.')
    parser.add_argument('--dataset', default=None,
                        help='The source dataset to use, e.g. cityscapes, voc.')
    parser.add_argument('--dataset-tgt', dest='dataset_tgt', default=None,
                        help='The target dataset to use, e.g. cityscapes, GM.')
    parser.add_argument('--split', dest='split', default='train',
                        help='The split to use, e.g. train, trainval.')
    parser.add_argument('--split-tgt', dest='split_tgt', default='val',
                        help='The split to use in target domain e.g. train, trainval.')
    parser.add_argument('--data-root', dest='data_root',
                        help='The root data dir. for source domain',
                        default=None, type=str)
    parser.add_argument('--data-root-tgt', dest='data_root_tgt',
                        help='The root data dir. for target domain',
                        default=None, type=str)
    parser.add_argument('--output', default=None,
                        help='The output dir.')
    parser.add_argument('--model', default=None,
                        help='The unique label of this model.')
    parser.add_argument('--batch-images', dest='batch_images',
                        help='The number of images per batch.',
                        default=None, type=int)
    parser.add_argument('--crop-size', dest='crop_size',
                        help='The size of network input during training.',
                        default=None, type=int)
    parser.add_argument('--origin-size', dest='origin_size',
                        help='The size of images to crop from in source domain',
                        default=2048, type=int)
    parser.add_argument('--origin-size-tgt', dest='origin_size_tgt',
                        help='The size of images to crop from in target domain',
                        default=2048, type=int)
    parser.add_argument('--scale-rate-range', dest='scale_rate_range',
                        help='The range of rescaling',
                        default='0.7,1.3', type=str)
    parser.add_argument('--weights', default=None,
                        help='The path of a pretrained model.')
    parser.add_argument('--gpus', default='0',
                        help='The devices to use, e.g. 0,1,2,3')
    #
    parser.add_argument('--lr-type', dest='lr_type',
                        help='The learning rate scheduler, e.g., fixed(default)/step/linear',
                        default=None, type=str)
    parser.add_argument('--base-lr', dest='base_lr',
                        help='The lr to start from.',
                        default=None, type=float)
    parser.add_argument('--lr-steps', dest='lr_steps',
                        help='The steps when to reduce lr.',
                        default=None, type=str)
    parser.add_argument('--weight-decay', dest='weight_decay',
                        help='The weight decay in sgd.',
                        default=0.0005, type=float)
    #
    parser.add_argument('--from-epoch', dest='from_epoch',
                        help='The epoch to start from.',
                        default=None, type=int)
    parser.add_argument('--stop-epoch', dest='stop_epoch',
                        help='The index of epoch to stop.',
                        default=None, type=int)
    parser.add_argument('--to-epoch', dest='to_epoch',
                        help='The number of epochs to run.',
                        default=None, type=int)
    # how many rounds to generate pseudo labels
    parser.add_argument('--idx-round', dest='idx_round',
                        help='The current number of rounds to generate pseudo labels',
                        default=0, type=int)

    # initial portion of selected pseudo labels in target domain
    parser.add_argument('--init-tgt-port', dest='init_tgt_port',
                        help='The initial portion of pixels selected in target dataset, both by global and class-wise threshold',
                        default=0.3, type=float)
    parser.add_argument('--init-src-port', dest='init_src_port',
                        help='The initial portion of images selected in source dataset',
                        default=0.3, type=float)
    parser.add_argument('--seed-int', dest='seed_int',
                        help='The random seed',
                        default=0, type=int)
    parser.add_argument('--mine-port', dest='mine_port',
                        help='The portion of data being mined',
                        default=0.5, type=float)
    #
    parser.add_argument('--mine-id-number', dest='mine_id_number',
                        help='Thresholding value for deciding mine id',
                        default=3, type=int)
    parser.add_argument('--mine-thresh', dest='mine_thresh',
                        help='The threshold to determine the mine id',
                        default=0.001, type=float)
    parser.add_argument('--mine-id-address', dest='mine_id_address',
                        help='The address of mine id',
                        default=None, type=str)
    #
    parser.add_argument('--phase',
                        help='Phase of this call, e.g., train/val.',
                        default='train', type=str)
    parser.add_argument('--with-prior', dest='with_prior',
                        help='with prior',
                        default='False', type=str)
    # for testing
    parser.add_argument('--test-scales', dest='test_scales',
                        help='Lengths of the longer side to resize an image into, e.g., 224,256.',
                        default=None, type=str)
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument('--test-steps', dest='test_steps',
                        help='The number of steps to take, for predictions at a higher resolution.',
                        default=1, type=int)
    #
    parser.add_argument('--kvstore', dest='kvstore',
                        help='The type of kvstore, e.g., local/device.',
                        default='local', type=str)
    parser.add_argument('--prefetch-threads', dest='prefetch_threads',
                        help='The number of threads to fetch data.',
                        default=1, type=int)
    parser.add_argument('--prefetcher', dest='prefetcher',
                        help='The type of prefetercher, e.g., process/thread.',
                        default='thread', type=str)
    parser.add_argument('--cache-images', dest='cache_images',
                        help='If cache images, e.g., 0/1',
                        default=None, type=int)
    parser.add_argument('--log-file', dest='log_file',
                        default=None, type=str)
    parser.add_argument('--check-start', dest='check_start',
                        help='The first epoch to snapshot.',
                        default=1, type=int)
    parser.add_argument('--check-step', dest='check_step',
                        help='The steps between adjacent snapshots.',
                        default=4, type=int)
    parser.add_argument('--debug',
                        help='True means logging debug info.',
                        default=False, action='store_true')
    parser.add_argument('--backward-do-mirror', dest='backward_do_mirror',
                        help='True means less gpu memory usage.',
                        default=False, action='store_true')
    parser.add_argument('--no-cudnn', dest='no_mxnet_cudnn_autotune_default',
                        help='True means deploy cudnn.',
                        default=False, action='store_true')
    parser.add_argument('--kc-policy', dest='kc_policy',
                        help='The kc determination policy, currently only "global" and "cb" (class-balanced)',
                        default='cb', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.debug:
        os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'

    if args.backward_do_mirror:
        os.environ['MXNET_BACKWARD_DO_MIRROR'] = '1'

    if args.no_mxnet_cudnn_autotune_default:
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

    if args.output is None:
        if args.phase == 'val':
            args.output = osp.dirname(args.weights)
        else:
            args.output = 'output'

    if args.weights is not None:
        if args.model is None:
            assert '_ep-' in args.weights
            parts = osp.basename(args.weights).split('_ep-')
            args.model = '_'.join(parts[:-1])
        if args.phase == 'train':
            if args.from_epoch is None:
                assert '_ep-' in args.weights
                parts = os.path.basename(args.weights).split('_ep-')
                assert len(parts) == 2
                from_model = parts[0]
                if from_model == args.model:
                    parts = os.path.splitext(os.path.basename(args.weights))[0].split('-')
                    args.from_epoch = int(parts[-1])

    if args.model is None:
        raise NotImplementedError('Missing argument: args.model')

    if args.from_epoch is None:
        args.from_epoch = 0

    if args.log_file is None:
        if args.phase == 'train':
            args.log_file = '{}.log'.format(args.model)
        elif args.phase == 'val':
            suffix = ''
            if args.split_tgt != 'val':
                suffix = '_{}'.format(args.split_tgt)
            args.log_file = '{}{}.log'.format(osp.splitext(osp.basename(args.weights))[0], suffix)
        else:
            raise NotImplementedError('Unknown phase: {}'.format(args.phase))
    model_specs = parse_model_label(args)
    if args.data_root is None:
        args.data_root = osp.join('data', model_specs['dataset'])

    return args, model_specs

def get_dataset_specs_tgt(args, model_specs):
    dataset = args.dataset
    dataset_tgt = args.dataset_tgt
    meta = {}
    meta_path = osp.join('issegm/data', dataset_tgt, 'meta.pkl')
    if osp.isfile(meta_path):
        with open(meta_path) as f:
            meta = cPickle.load(f)

    mine_id = None
    mine_id_priority = None
    mine_port = args.mine_port
    mine_th = args.mine_thresh
    cmap_path = 'data/shared/cmap.pkl'
    cache_images = args.phase == 'train'
    mx_workspace = 1650
    sys.path.insert(0, 'data/cityscapesscripts/helpers')
    if args.phase == 'train':
        mine_id = np.load(args.mine_id_address + '/mine_id.npy')
        mine_id_priority = np.load(args.mine_id_address + '/mine_id_priority.npy')
        mine_th = np.zeros(len(mine_id))  # trainId starts from 0
    if dataset == 'gta' and dataset_tgt == 'cityscapes':
        from labels import id2label, trainId2label
        #
        label_2_id_tgt = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id_tgt[l] = id2label[l].trainId
        id_2_label_tgt = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels_tgt = sorted(set(id_2_label_tgt.ravel()))
        id_2_label_src = id_2_label_tgt
        label_2_id_src = label_2_id_tgt
        valid_labels_src = valid_labels_tgt
        #
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        # max_shape_src = np.array((1052, 1914))
        max_shape_src = np.array((1024, 2048))
        max_shape_tgt = np.array((1024, 2048))
        #
        if args.split in ('train+', 'trainval+'):
            cache_images = False
        #
        if args.phase in ('val',):
            mx_workspace = 8192
    elif dataset == 'synthia' and dataset_tgt == 'cityscapes':
        from labels_cityscapes_synthia import id2label as id2label_tgt
        from labels_cityscapes_synthia import trainId2label as trainId2label_tgt
        from labels_synthia import id2label as id2label_src
        label_2_id_src = 255 * np.ones((256,))
        for l in id2label_src:
            if l in (-1, 255):
                continue
            label_2_id_src[l] = id2label_src[l].trainId
        label_2_id_tgt = 255 * np.ones((256,))
        for l in id2label_tgt:
            if l in (-1, 255):
                continue
            label_2_id_tgt[l] = id2label_tgt[l].trainId
        id_2_label_tgt = np.array([trainId2label_tgt[_].id for _ in trainId2label_tgt if _ not in (-1, 255)])
        valid_labels_tgt = sorted(set(id_2_label_tgt.ravel()))
        id_2_label_src = None
        valid_labels_src = None
        #
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in id2label_tgt.keys():
            cmap[i] = id2label_tgt[i].color
        #
        ident_size = True
        #
        max_shape_src = np.array((760, 1280))
        max_shape_tgt = np.array((1024, 2048))
        #
        if args.split in ('train+', 'trainval+'):
            cache_images = False
        #
        if args.phase in ('val',):
            mx_workspace = 8192
    else:
        raise NotImplementedError('Unknow dataset: {}'.format(args.dataset))

    if cmap is None and cmap_path is not None:
        if osp.isfile(cmap_path):
            with open(cmap_path) as f:
                cmap = cPickle.load(f)

    meta['gpus'] = args.gpus
    meta['mine_port'] = mine_port
    meta['mine_id'] = mine_id
    meta['mine_id_priority'] = mine_id_priority
    meta['mine_th'] = mine_th
    meta['label_2_id_tgt'] = label_2_id_tgt
    meta['id_2_label_tgt'] = id_2_label_tgt
    meta['valid_labels_tgt'] = valid_labels_tgt
    meta['label_2_id_src'] = label_2_id_src
    meta['id_2_label_src'] = id_2_label_src
    meta['valid_labels_src'] = valid_labels_src
    meta['cmap'] = cmap
    meta['ident_size'] = ident_size
    meta['max_shape_src'] = meta.get('max_shape_src', max_shape_src)
    meta['max_shape_tgt'] = meta.get('max_shape_tgt', max_shape_tgt)
    meta['cache_images'] = args.cache_images if args.cache_images is not None else cache_images
    meta['mx_workspace'] = mx_workspace
    return meta

def _get_metric():
    def _eval_func(label, pred):
        # global sxloss
        gt_label = label.ravel()
        valid_flag = gt_label != 255
        labels = gt_label[valid_flag].astype(int)
        n,c,h,w = pred.shape
        valid_inds = np.where(valid_flag)[0]
        probmap = np.rollaxis(pred.astype(np.float32),1).reshape((c, -1))
        valid_probmap = probmap[labels, valid_inds]
        log_valid_probmap = -np.log(valid_probmap+1e-32)
        sum_metric = log_valid_probmap.sum()
        num_inst = valid_flag.sum()

        return (sum_metric, num_inst + (num_inst == 0))

    return mx.metric.CustomMetric(_eval_func, 'loss')

def _get_scalemeanstd():
    if model_specs['net_type'] in ('rna',):
        return (1.0 / 255,
                np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),
                np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)))
    return None, None, None

def _get_transformer_image():
    scale, mean_, std_ = _get_scalemeanstd()
    transformers = []
    if scale > 0:
        transformers.append(ts.ColorScale(np.single(scale)))
    transformers.append(ts.ColorNormalize(mean_, std_))
    return transformers

def _get_module(args, margs, dargs, net=None):
    if net is None:
        # the following lines show how to create symbols for our networks
        if model_specs['net_type'] == 'rna':
            from util.symbol.symbol import cfg as symcfg
            symcfg['lr_type'] = 'alex'
            symcfg['workspace'] = dargs.mx_workspace
            symcfg['bn_use_global_stats'] = True
            if model_specs['net_name'] == 'a1':
                from util.symbol.resnet_v2 import fcrna_model_a1
                net = fcrna_model_a1(margs.classes, margs.feat_stride, bootstrapping=False)
        if net is None:
            raise NotImplementedError('Unknown network: {}'.format(vars(margs)))
    contexts = [mx.gpu(int(_)) for _ in args.gpus.split(',')]
    mod = mx.mod.Module(net, context=contexts)
    return mod

def _make_dirs(path):
    if not osp.isdir(path):
        os.makedirs(path)

def facc(label, pred):
    pred = pred.argmax(1).ravel()
    label = label.ravel()
    return (pred == label).mean()

def fentropy(label, pred):
    pred_source = pred[:, 1, :, :].ravel()
    label = label.ravel()
    return -(label * np.log(pred_source + 1e-12) + (1. - label) * np.log(1. - pred_source + 1e-12)).mean()

def _interp_preds_as_impl(num_classes, im_size, pred_stride, imh, imw, pred):
    imh0, imw0 = im_size
    pred = pred.astype(np.single, copy=False)
    input_h, input_w = pred.shape[0] * pred_stride, pred.shape[1] * pred_stride
    assert pred_stride >= 1.
    this_interp_pred = np.array(Image.fromarray(pred).resize((input_w, input_h), Image.CUBIC))
    if imh0 == imh:
        interp_pred = this_interp_pred[:imh, :imw]
    else:
        interp_method = util.get_interp_method(imh, imw, imh0, imw0)
        interp_pred = np.array(Image.fromarray(this_interp_pred[:imh, :imw]).resize((imw0, imh0), interp_method))
    return interp_pred


def interp_preds_as(im_size, net_preds, pred_stride, imh, imw, threads=4):
    num_classes = net_preds.shape[0]
    worker = partial(_interp_preds_as_impl, num_classes, im_size, pred_stride, imh, imw)
    if threads == 1:
        ret = [worker(_) for _ in net_preds]
    else:
        pool = Pool(threads)
        ret = pool.map(worker, net_preds)
        pool.close()
    return np.array(ret)


class ScoreUpdater(object):
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels

        self._confs = np.zeros((c_num, c_num, x_num))
        self._pixels = np.zeros((c_num, x_num))
        self._logger = logger
        self._label = label
        self._info = info

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros((self._pixels.shape[1],))
        self._confs[:] = 0
        self._pixels[:] = 0

    @staticmethod
    def calc_updates(valid_labels, pred_label, label):
        num_classes = len(valid_labels)

        pred_flags = [set(np.where((pred_label == _).ravel())[0]) for _ in valid_labels]
        class_flags = [set(np.where((label == _).ravel())[0]) for _ in valid_labels]

        conf = [len(class_flags[j].intersection(pred_flags[k])) for j in xrange(num_classes) for k in
                xrange(num_classes)]
        pixel = [len(class_flags[j]) for j in xrange(num_classes)]
        return np.single(conf).reshape((num_classes, num_classes)), np.single(pixel)

    def do_updates(self, conf, pixel, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._confs[:, :, i] = conf
        self._pixels[:, i] = pixel

    def update(self, pred_label, label, i, computed=True):
        conf, pixel = ScoreUpdater.calc_updates(self._valid_labels, pred_label, label)
        self.do_updates(conf, pixel, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        confs = self._confs
        pixels = self._pixels

        num_classes = pixels.shape[0]
        x_num = pixels.shape[1]

        class_pixels = pixels.sum(1)
        class_pixels += class_pixels == 0
        scores = confs[xrange(num_classes), xrange(num_classes), :].sum(1)
        acc = scores.sum() / pixels.sum()
        cls_accs = scores / class_pixels
        class_preds = confs.sum(0).sum(1)
        ious = scores / (class_pixels + class_preds - scores)

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}pixel acc: {:.2f}%, mean acc: {:.2f}%, mean iou: {:.2f}%'. \
                        format(name, acc * 100, cls_accs.mean() * 100, ious.mean() * 100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(cls_accs * 100))
                logger.info('\n{}'.format(ious * 100))

        return acc, cls_accs, ious

    def overall_scores(self, logger=None):
        acc, cls_accs, ious = self.scores(None, logger)
        return acc, cls_accs.mean(), ious.mean()

def _train_impl(args, model_specs, logger):
    if len(args.output) > 0:
        _make_dirs(args.output)
    # dataiter
    dataset_specs_tgt = get_dataset_specs_tgt(args, model_specs)
    scale, mean_, _ = _get_scalemeanstd()
    if scale > 0:
        mean_ /= scale
    margs = argparse.Namespace(**model_specs)
    dargs = argparse.Namespace(**dataset_specs_tgt)
    # number of list_lines
    split_filename = 'issegm/data/{}/{}.lst'.format(margs.dataset, args.split)
    num_source = 0
    with open(split_filename) as f:
        for item in f.readlines():
            num_source = num_source + 1
    #
    batches_per_epoch = num_source // args.batch_images
    # optimizer
    assert args.to_epoch is not None
    if args.stop_epoch is not None:
        assert args.stop_epoch > args.from_epoch and args.stop_epoch <= args.to_epoch
    else:
        args.stop_epoch = args.to_epoch

    from_iter = args.from_epoch * batches_per_epoch
    to_iter = args.to_epoch * batches_per_epoch
    lr_params = model_specs['lr_params']
    base_lr = lr_params['base']
    if lr_params['type'] == 'fixed':
        scheduler = FixedScheduler()
    elif lr_params['type'] == 'step':
        left_step = []
        for step in lr_params['args']['step']:
            if from_iter > step:
                base_lr *= lr_params['args']['factor']
                continue
            left_step.append(step - from_iter)
        model_specs['lr_params']['step'] = left_step
        scheduler = mx.lr_scheduler.MultiFactorScheduler(**lr_params['args'])
    elif lr_params['type'] == 'linear':
        scheduler = LinearScheduler(updates=to_iter + 1, frequency=50,
                                    stop_lr=min(base_lr / 100., 1e-6),
                                    offset=from_iter)
    elif lr_params['type'] == 'poly':
        scheduler = PolyScheduler(updates=to_iter + 1, frequency=50,
                                  stop_lr=min(base_lr / 100., 1e-8),
                                  power=0.9,
                                  offset=from_iter)

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    optimizer_params = {
        'learning_rate': base_lr,
        'momentum': 0.9,
        'wd': args.weight_decay,
        'lr_scheduler': scheduler,
        'rescale_grad': 1.0 / len(args.gpus.split(',')),
    }

    data_src_port = args.init_src_port
    data_src_num = int(num_source * data_src_port)
    mod = _get_module(args, margs, dargs)
    addr_weights = args.weights  # first weights should be xxxx_ep-0000.params!
    addr_output = args.output

    # initializer
    net_args = None
    net_auxs = None
    ###
    if addr_weights is not None:
        net_args, net_auxs = mxutil.load_params_from_file(addr_weights)

    ####################################### training model
    to_model = osp.join(addr_output, str(args.idx_round), '{}_ep'.format(args.model))
    dataiter = FileIter(dataset=margs.dataset,
                        split=args.split,
                        data_root=args.data_root,
                        num_sel_source=data_src_num,
                        num_source=num_source,
                        seed_int=args.seed_int,
                        dataset_tgt=args.dataset_tgt,
                        split_tgt=args.split_tgt,
                        data_root_tgt=args.data_root_tgt,
                        sampler='random',
                        batch_images=args.batch_images,
                        meta=dataset_specs_tgt,
                        rgb_mean=mean_,
                        feat_stride=margs.feat_stride,
                        label_stride=margs.feat_stride,
                        origin_size=args.origin_size,
                        origin_size_tgt=args.origin_size_tgt,
                        crop_size=args.crop_size,
                        scale_rate_range=[float(_) for _ in args.scale_rate_range.split(',')],
                        transformer=None,
                        transformer_image=ts.Compose(_get_transformer_image()),
                        prefetch_threads=args.prefetch_threads,
                        prefetcher_type=args.prefetcher,
                        )
    dataiter.reset()
    mod.fit(
        dataiter,
        eval_metric=_get_metric(),
        batch_end_callback=mx.callback.log_train_metric(10, auto_reset=False),
        epoch_end_callback=mx.callback.do_checkpoint(to_model),
        kvstore=args.kvstore,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        initializer=initializer,
        arg_params=net_args,
        aux_params=net_auxs,
        allow_missing=args.from_epoch == 0,
        begin_epoch=args.from_epoch,
        num_epoch=args.stop_epoch,
    )

# @profile
# MST:
def _val_impl(args, model_specs, logger):
    if len(args.output) > 0:
        _make_dirs(args.output)
    # dataiter
    dataset_specs_tgt = get_dataset_specs_tgt(args, model_specs)
    scale, mean_, _ = _get_scalemeanstd()
    if scale > 0:
        mean_ /= scale
    margs = argparse.Namespace(**model_specs)
    dargs = argparse.Namespace(**dataset_specs_tgt)
    mod = _get_module(args, margs, dargs)
    addr_weights = args.weights # first weights should be xxxx_ep-0000.params!
    addr_output = args.output
    # current round index
    cround = args.idx_round

    net_args = None
    net_auxs = None
    ###
    if addr_weights is not None:
        net_args, net_auxs = mxutil.load_params_from_file(addr_weights)
    ######
    save_dir = osp.join(args.output, str(cround), osp.splitext(args.log_file)[0])
    # pseudo labels
    save_dir_pseudo_labelIds = osp.join(save_dir, 'pseudo_labelIds')
    save_dir_pseudo_color = osp.join(save_dir, 'pseudo_color')
    # without sp
    save_dir_nplabelIds = osp.join(save_dir, 'nplabelIds')
    save_dir_npcolor = osp.join(save_dir, 'npcolor')
    # probability map
    save_dir_probmap = osp.join(args.output, 'probmap')
    save_dir_stats = osp.join(args.output, 'stats')

    _make_dirs(save_dir)
    _make_dirs(save_dir_pseudo_labelIds)
    _make_dirs(save_dir_pseudo_color)
    _make_dirs(save_dir_nplabelIds)
    _make_dirs(save_dir_npcolor)
    _make_dirs(save_dir_probmap)
    _make_dirs(save_dir_stats)
    if args.with_prior == 'True':
        # with sp
        save_dir_splabelIds = osp.join(save_dir, 'splabelIds')
        save_dir_spcolor = osp.join(save_dir, 'spcolor')
        _make_dirs(save_dir_splabelIds)
        _make_dirs(save_dir_spcolor)
    if args.kc_policy == 'cb':
        # reweighted prediction map
        save_dir_rwlabelIds = osp.join(save_dir, 'rwlabelIds')
        save_dir_rwcolor = osp.join(save_dir, 'rwcolor')
        _make_dirs(save_dir_rwlabelIds)
        _make_dirs(save_dir_rwcolor)
    ######
    dataset_tgt = model_specs['dataset_tgt']
    image_list_tgt, label_gt_list_tgt,image_tgt_list = parse_split_file_tgt(margs.dataset_tgt, args.split_tgt)
    has_gt = args.split_tgt in ('train', 'val',)
    crop_sizes = sorted([int(_) for _ in args.test_scales.split(',')])[::-1]
    crop_size = crop_sizes[0]
    assert len(crop_sizes) == 1, 'multi-scale testing not implemented'
    label_stride = margs.feat_stride
    x_num = len(image_list_tgt)
    do_forward = True
    # for all images that has the same resolution
    if do_forward:
        batch = None
        transformers = [ts.Scale(crop_size, Image.CUBIC, False)]
        transformers += _get_transformer_image()
        transformer = ts.Compose(transformers)

    scorer_np = ScoreUpdater(dargs.valid_labels_tgt, margs.classes, x_num, logger)
    scorer_np.reset()
    # with prior
    if args.with_prior == 'True':
        scorer = ScoreUpdater(dargs.valid_labels_tgt, margs.classes, x_num, logger)
        scorer.reset()

    done_count = 0 # for multi-scale testing
    num_classes = margs.classes
    init_tgt_port = float(args.init_tgt_port)
    # class-wise
    cls_exist_array = np.zeros([1, num_classes], dtype=int)
    cls_thresh = np.zeros([num_classes]) # confidence thresholds for all classes
    cls_size = np.zeros([num_classes]) # number of predictions in each class
    array_pixel = 0.0
    # prior
    if args.with_prior == 'True':
        in_path_prior = 'spatial_prior/{}/prior_array.mat'.format(args.dataset)
        sprior = scipy.io.loadmat(in_path_prior)
        prior_array = sprior["prior_array"].astype(np.float32)
        #prior_array = np.maximum(prior_array,0)
    ############################ network forward
    for i in xrange(x_num):
        start = time.time()
        ############################ network forward on single image (from official ResNet-38 implementation)
        sample_name = osp.splitext(osp.basename(image_list_tgt[i]))[0]
        im_path = osp.join(args.data_root_tgt, image_list_tgt[i])
        rim = np.array(Image.open(im_path).convert('RGB'), np.uint8)
        if do_forward:
            im = transformer(rim)
            imh, imw = im.shape[:2]
            # init
            if batch is None:
                if dargs.ident_size:
                    input_h = make_divisible(imh, margs.feat_stride)
                    input_w = make_divisible(imw, margs.feat_stride)
                else:
                    input_h = input_w = make_divisible(crop_size, margs.feat_stride)
                label_h, label_w = input_h / label_stride, input_w / label_stride
                test_steps = args.test_steps
                pred_stride = label_stride / test_steps
                pred_h, pred_w = label_h * test_steps, label_w * test_steps
                input_data = np.zeros((1, 3, input_h, input_w), np.single)
                input_label = 255 * np.ones((1, label_h * label_w), np.single)
                dataiter_tgt = mx.io.NDArrayIter(input_data, input_label)
                batch = dataiter_tgt.next()

                mod.bind(dataiter_tgt.provide_data, dataiter_tgt.provide_label, for_training=False, force_rebind=True)
                if not mod.params_initialized:
                    mod.init_params(arg_params=net_args, aux_params=net_auxs)

            nim = np.zeros((3, imh + label_stride, imw + label_stride), np.single)
            sy = sx = label_stride // 2
            nim[:, sy:sy + imh, sx:sx + imw] = im.transpose(2, 0, 1)

            net_preds = np.zeros((margs.classes, pred_h, pred_w), np.single)
            sy = sx = pred_stride // 2 + np.arange(test_steps) * pred_stride
            for ix in xrange(test_steps):
                for iy in xrange(test_steps):
                    input_data = np.zeros((1, 3, input_h, input_w), np.single)
                    input_data[0, :, :imh, :imw] = nim[:, sy[iy]:sy[iy] + imh, sx[ix]:sx[ix] + imw]
                    batch.data[0] = mx.nd.array(input_data)
                    mod.forward(batch, is_train=False)
                    this_call_preds = mod.get_outputs()[0].asnumpy()[0]
                    if args.test_flipping:
                        batch.data[0] = mx.nd.array(input_data[:, :, :, ::-1])
                        mod.forward(batch, is_train=False)
                        # average the original and flipped image prediction
                        this_call_preds = 0.5 * (
                        this_call_preds + mod.get_outputs()[0].asnumpy()[0][:, :, ::-1])
                    net_preds[:, iy:iy + pred_h:test_steps, ix:ix + pred_w:test_steps] = this_call_preds
        interp_preds_np = interp_preds_as(rim.shape[:2], net_preds, pred_stride, imh, imw)
        ########################### #save predicted labels and confidence score vectors in target domains
        # no prior prediction with trainIDs
        pred_label_np = interp_preds_np.argmax(0)
        # no prior prediction with labelIDs
        if dargs.id_2_label_tgt is not None:
            pred_label_np = dargs.id_2_label_tgt[pred_label_np]
        # no prior color prediction
        im_to_save_np = Image.fromarray(pred_label_np.astype(np.uint8))
        im_to_save_npcolor = im_to_save_np.copy()
        if dargs.cmap is not None:
            im_to_save_npcolor.putpalette(dargs.cmap.ravel())
        # save no prior prediction with labelIDs and colors
        out_path_np = osp.join(save_dir_nplabelIds, '{}.png'.format(sample_name))
        out_path_npcolor = osp.join(save_dir_npcolor, '{}.png'.format(sample_name))
        im_to_save_np.save(out_path_np)
        im_to_save_npcolor.save(out_path_npcolor)
        # with prior
        if args.with_prior == 'True':
            probmap = np.multiply(prior_array,interp_preds_np).astype(np.float32)
        elif args.with_prior == 'False':
            probmap = interp_preds_np.copy().astype(np.float32)
        pred_label = probmap.argmax(0)
        probmap_max = np.amax(probmap, axis=0)
        ############################ save confidence scores of target domain as class-wise vectors
        for idx_cls in np.arange(0, num_classes):
            idx_temp = pred_label == idx_cls
            sname = 'array_cls' + str(idx_cls)
            if not (sname in locals()):
                exec ("%s = np.float32(0)" % sname)
            if idx_temp.any():
                cls_exist_array[0, idx_cls] = 1
                probmap_max_cls_temp = probmap_max[idx_temp].astype(np.float32)
                len_cls = probmap_max_cls_temp.size
                # downsampling by rate 4
                probmap_cls = probmap_max_cls_temp[0:len_cls]
                exec ("%s = np.append(%s,probmap_cls)" % (sname, sname))
        ############################ save prediction
        # save prediction probablity map
        out_path_probmap = osp.join(save_dir_probmap, '{}.npy'.format(sample_name))
        np.save(out_path_probmap, probmap.astype(np.float32))
        # save predictions with spatial priors, if sp exist.
        if args.with_prior == 'True':
            if dargs.id_2_label_tgt is not None:
                pred_label = dargs.id_2_label_tgt[pred_label]
            im_to_save_sp = Image.fromarray(pred_label.astype(np.uint8))
            im_to_save_spcolor = im_to_save_sp.copy()
            if dargs.cmap is not None:  # save color seg map
                im_to_save_spcolor.putpalette(dargs.cmap.ravel())
            out_path_sp = osp.join(save_dir_splabelIds, '{}.png'.format(sample_name))
            out_path_spcolor = osp.join(save_dir_spcolor, '{}.png'.format(sample_name))
            im_to_save_sp.save(out_path_sp)
            im_to_save_spcolor.save(out_path_spcolor)
        # log information
        done_count += 1
        if not has_gt:
            logger.info(
                'Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, 1. * done_count / (time.time() - start)))
            continue
        if args.split_tgt in ('train', 'val'):
            # evaluate with ground truth
            label_path = osp.join(args.data_root_tgt, label_gt_list_tgt[i])
            label = np.array(Image.open(label_path), np.uint8)
            if args.with_prior == 'True':
                scorer.update(pred_label, label, i)
            scorer_np.update(pred_label_np, label, i)
    # save target training list
    fout = 'issegm/data/{}/{}_training_gpu{}.lst'.format(args.dataset_tgt,args.split_tgt,args.gpus)
    fo = open(fout, "w")
    for idx_image in range(x_num):
        sample_name = osp.splitext(osp.basename(image_list_tgt[idx_image]))[0]
        fo.write(image_tgt_list[idx_image] + '\t' + osp.join(save_dir_pseudo_labelIds, '{}.png'.format(sample_name)) + '\n')
    fo.close()
    ############################ kc generation
    start_sort = time.time()
    # threshold for each class
    if args.kc_policy == 'global':
        for idx_cls in np.arange(0,num_classes):
            tname = 'array_cls' + str(idx_cls)
            exec ("array_pixel = np.append(array_pixel,%s)" % tname) # reverse=False for ascending losses and reverse=True for descending confidence
        array_pixel = sorted(array_pixel, reverse = True)
        len_cls = len(array_pixel)
        len_thresh = int(math.floor(len_cls * init_tgt_port))
        cls_size[:] = len_cls
        cls_thresh[:] = array_pixel[len_thresh-1].copy()
        array_pixel = 0.0
    if args.kc_policy == 'cb':
        for idx_cls in np.arange(0, num_classes):
            tname = 'array_cls' + str(idx_cls)
            if cls_exist_array[0, idx_cls] == 1:
                exec("%s = sorted(%s,reverse=True)" % (tname, tname)) # reverse=False for ascending losses and reverse=True for descending confidence
                exec("len_cls = len(%s)" % tname)
                cls_size[idx_cls] = len_cls
                len_thresh = int(math.floor(len_cls * init_tgt_port))
                if len_thresh != 0:
                    exec("cls_thresh[idx_cls] = %s[len_thresh-1].copy()" % tname)
                exec("%s = %d" % (tname, 0.0))

    # threshold for mine_id with priority
    mine_id_priority = np.nonzero(cls_size / np.sum(cls_size) < args.mine_thresh)[0]
    # chosen mine_id
    mine_id_all = np.argsort(cls_size / np.sum(cls_size))
    mine_id = mine_id_all[:args.mine_id_number]
    print(mine_id)
    np.save(save_dir_stats + '/mine_id.npy', mine_id)
    np.save(save_dir_stats + '/mine_id_priority.npy', mine_id_priority)
    np.save(save_dir_stats + '/cls_thresh.npy', cls_thresh)
    np.save(save_dir_stats + '/cls_size.npy', cls_size)
    logger.info('Kc determination done in %.2f s.', time.time() - start_sort)
    ############################ pseudo-label generation 
    for i in xrange(x_num):
        sample_name = osp.splitext(osp.basename(image_list_tgt[i]))[0]
        sample_pseudo_label_name = osp.join(save_dir_pseudo_labelIds, '{}.png'.format(sample_name))
        sample_pseudocolor_label_name = osp.join(save_dir_pseudo_color, '{}.png'.format(sample_name))
        out_path_probmap = osp.join(save_dir_probmap, '{}.npy'.format(sample_name))
        probmap = np.load(out_path_probmap)
        rw_probmap = np.zeros(probmap.shape, np.single)
        cls_thresh[cls_thresh == 0] = 1.0 # cls_thresh = 0 means there is no prediction in this class
        ############# pseudo-label assignment
        for idx_cls in np.arange(0, num_classes):
            cls_thresh_temp = cls_thresh[idx_cls]
            cls_probmap = probmap[idx_cls,:,:]
            cls_rw_probmap = np.true_divide(cls_probmap,cls_thresh_temp)
            rw_probmap[idx_cls,:,:] = cls_rw_probmap.copy()

        rw_probmap_max = np.amax(rw_probmap, axis=0)
        pseudo_label = np.argmax(rw_probmap,axis=0)
        ############# pseudo-label selection
        idx_unconfid = rw_probmap_max < 1
        idx_confid = rw_probmap_max >= 1
        # pseudo-labels with labelID
        pseudo_label = pseudo_label.astype(np.uint8)
        pseudo_label_labelID = dargs.id_2_label_tgt[pseudo_label]
        rw_pred_label = pseudo_label_labelID.copy()
        # ignore label assignment, compatible with labelIDs
        pseudo_label_labelID[idx_unconfid] = 0
        ############# save pseudo-label
        im_to_save_pseudo = Image.fromarray(pseudo_label_labelID.astype(np.uint8))
        im_to_save_pseudocol = im_to_save_pseudo.copy()
        if dargs.cmap is not None:  # save segmentation prediction with color
            im_to_save_pseudocol.putpalette(dargs.cmap.ravel())
        out_path_pseudo = osp.join(save_dir_pseudo_labelIds, '{}.png'.format(sample_name))
        out_path_colpseudo = osp.join(save_dir_pseudo_color, '{}.png'.format(sample_name))
        im_to_save_pseudo.save(out_path_pseudo)
        im_to_save_pseudocol.save(out_path_colpseudo)
        ############# save reweighted pseudo-label in cbst
        if args.kc_policy == 'cb':
            im_to_save_rw = Image.fromarray(rw_pred_label.astype(np.uint8))
            im_to_save_rwcolor = im_to_save_rw.copy()
            if dargs.cmap is not None:
                im_to_save_rwcolor.putpalette(dargs.cmap.ravel())
            out_path_rw = osp.join(save_dir_rwlabelIds, '{}.png'.format(sample_name))
            out_path_rwcolor = osp.join(save_dir_rwcolor, '{}.png'.format(sample_name))
            # save no prior prediction with labelIDs and colors
            im_to_save_rw.save(out_path_rw)
            im_to_save_rwcolor.save(out_path_rwcolor)

    ## remove probmap folder
    import shutil
    shutil.rmtree(save_dir_probmap)
    ##

if __name__ == "__main__":
    util.cfg['choose_interpolation_method'] = True

    args, model_specs = parse_args()

    if len(args.output) > 0:
        _make_dirs(args.output)

    logger = util.set_logger(args.output, args.log_file, args.debug)
    logger.info('start with arguments %s', args)
    logger.info('and model specs %s', model_specs)

    if args.phase == 'train':
        _train_impl(args, model_specs, logger)
    elif args.phase == 'val':
        _val_impl(args, model_specs, logger)
    else:
        raise NotImplementedError('Unknown phase: {}'.format(args.phase))
