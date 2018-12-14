# pylint: skip-file
import argparse
import cPickle
import os
import os.path as osp
import copy
import re
import sys
import time
from functools import partial
from PIL import Image
from multiprocessing import Pool

import numpy as np

import mxnet as mx

from util import mxutil
from util import transformer as ts
from util import util
from util.lr_scheduler import FixedScheduler, LinearScheduler, PolyScheduler

from data_src import FileIter, make_divisible, parse_split_file


def parse_split_file_test(dataset, split, data_root=''):
    split_filename = 'issegm/data_list/{}/{}.lst'.format(dataset, split)
    image_list = []
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
    return image_list

def parse_model_label(args):
    assert args.model is not None
    fields = [_.strip() for _ in osp.basename(args.model).split('_')]
    # parse fields
    i = 0
    num_fields = len(fields)
    # database
    dataset = fields[i] if args.dataset is None else args.dataset
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
    # linear
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
    }
    return model_specs


def parse_args():
    parser = argparse.ArgumentParser(description='Tune FCRNs from ResNets.')
    parser.add_argument('--gpus', default='0',
                        help='The devices to use, e.g. 0,1,2,3')
    parser.add_argument('--dataset', default=None,
                        help='The dataset to use, e.g. cityscapes, voc.')
    parser.add_argument('--split', default='train',
                        help='The split to use, e.g. train, trainval.')
    parser.add_argument('--data-root', dest='data_root',
                        help='The root data dir.',
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
                        help='The size of images to crop from.',
                        default=None, type=int)
    parser.add_argument('--scale-rate-range', dest='scale_rate_range',
                        help='The range of rescaling',
                        default='0.7,1.3', type=str)
    parser.add_argument('--weights', default=None,
                        help='The path of a pretrained model.')
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
    #
    parser.add_argument('--phase',
                        help='Phase of this call, e.g., train/val.',
                        default='train', type=str)
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
    parser.add_argument('--save-predictions', dest='save_predictions', 
                        help='If save the predicted score maps.',
                        default=False, action='store_true')
    parser.add_argument('--no-save-results', dest='save_results', 
                        help='If save the predicted pixel-wise labels.',
                        default=True, action='store_false')
    #
    parser.add_argument('--kvstore', dest='kvstore',
                        help='The type of kvstore, e.g., local/device.',
                        default='device', type=str)
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
        #
        if args.model is None:
            assert '_ep-' in args.weights
            parts = osp.basename(args.weights).split('_ep-')
            args.model = '_'.join(parts[:-1])
        #
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
            if args.split != 'val':
                suffix = '_{}'.format(args.split)
            args.log_file = '{}{}.log'.format(osp.splitext(osp.basename(args.weights))[0], suffix)
        elif args.phase == 'test':
            args.log_file = '{}.log'.format(args.model)
        else:
            raise NotImplementedError('Unknown phase: {}'.format(args.phase))
    
    model_specs = parse_model_label(args)
    if args.data_root is None:
        args.data_root = osp.join('data', model_specs['dataset'])
    
    return args, model_specs


def get_dataset_specs(args, model_specs):
    dataset = model_specs['dataset']
    meta = {}
    meta_path = osp.join('issegm/data', dataset, 'meta.pkl')
    if osp.isfile(meta_path):
        with open(meta_path) as f:
            meta = cPickle.load(f)

    label_2_id = None
    id_2_label = None
    ident_size = False
    cmap = None
    cmap_path = 'data/shared/cmap.pkl'
    cache_images = args.phase == 'train'
    mx_workspace = 1650
    if dataset == 'cityscapes':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels = sorted(set(id_2_label.ravel()))
        #
        cmap = np.zeros((256,3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        max_shape = np.array((1024, 2048))
        #
        if args.split in ('train+', 'trainval+'):
            cache_images = False
        #
        if args.phase in ('val',):
            mx_workspace = 8192
    elif dataset == 'gta':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels = sorted(set(id_2_label.ravel()))
        #
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        max_shape = np.array((1052, 1914))
        #
        if args.phase in ('val',):
            mx_workspace = 8192
    elif dataset == 'synthia':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels_synthia import id2label, trainId2label
        #
        label_2_id = 255 * np.ones((256,))
        for l in id2label:
            if l in (-1, 255):
                continue
            label_2_id[l] = id2label[l].trainId
        id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
        valid_labels = sorted(set(id_2_label.ravel()))
        #
        cmap = np.zeros((256, 3), dtype=np.uint8)
        for i in id2label.keys():
            cmap[i] = id2label[i].color
        #
        ident_size = True
        #
        max_shape = np.array((760, 1280))
        #
        if args.phase in ('val',):
            mx_workspace = 8192

    else:
        raise NotImplementedError('Unknow dataset: {}'.format(dataset))
    
    if cmap is None and cmap_path is not None:
        if osp.isfile(cmap_path):
            with open(cmap_path) as f:
                cmap = cPickle.load(f)
    
    meta['label_2_id'] = label_2_id
    meta['id_2_label'] = id_2_label
    meta['valid_labels'] = valid_labels
    meta['cmap'] = cmap
    meta['ident_size'] = ident_size
    meta['max_shape'] = meta.get('max_shape', max_shape)
    meta['cache_images'] = args.cache_images if args.cache_images is not None else cache_images
    meta['mx_workspace'] = mx_workspace
    return meta


def _get_metric():
    def _eval_func(label, pred):
        # global sxloss
        gt_label = label.ravel()
        valid_flag = gt_label != 255
        # valid_flag = gt_label != 65
        gt_label = gt_label[valid_flag]
        n,c,h,w, = pred.shape
        gt_index = label.reshape(n,h,w).astype(int)
        index_flag = gt_index == 255
        # index_flag = gt_index == 65
        gt_index[index_flag] = 0
        pred1 = np.swapaxes(pred,1,2)
        pred2 = np.swapaxes(pred1,2,3)
        # temp = pred.sum(1).ravel()[valid_flag].sum() # this is equal to gt_label.size, which indicates the output of softmaxoutput layer is the softmax probability
        xv, yv, zv = np.meshgrid(np.arange(0,n), np.arange(0,h), np.arange(0,w), indexing='ij')
        gt_softmax = pred2[xv, yv, zv, gt_index].ravel()[valid_flag]
        softmax_output = gt_softmax # + gt_label
        sum_metric = np.log(softmax_output + 1e-6).sum()
        sum_metric = -sum_metric
        num_inst = valid_flag.sum()
        # sxloss = sum_metric/(num_inst + (num_inst == 0))
        return (sum_metric, num_inst + (num_inst == 0))
    return mx.metric.CustomMetric(_eval_func, 'loss')

def _get_scalemeanstd():
    if model_specs['net_type'] == 'rn':
        return -1, np.array([123.68, 116.779, 103.939]).reshape((1, 1, 3)), None
    if model_specs['net_type'] in ('rna',):
        return (1.0/255,
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
            if model_specs['net_name'] == 'd':
            # load network
                from importlib import import_module
                sym = import_module('util.symbol.resnet')
                net = sym.get_symbol(19, 101, '3,512,512', conv_workspace=1650)

        if net is None:
            raise NotImplementedError('Unknown network: {}'.format(vars(margs)))
    contexts = [mx.gpu(int(_)) for _ in args.gpus.split(',')]
    mod = mx.mod.Module(net, context=contexts)
    return mod

def _make_dirs(path):
    if not osp.isdir(path):
        os.makedirs(path)

def _train_impl(args, model_specs, logger):
    if len(args.output) > 0:
        _make_dirs(args.output)
    # dataiter
    dataset_specs = get_dataset_specs(args, model_specs)
    scale, mean_, _ = _get_scalemeanstd()
    if scale > 0:
        mean_ /= scale
    margs = argparse.Namespace(**model_specs)
    dargs = argparse.Namespace(**dataset_specs)
    dataiter = FileIter(dataset=margs.dataset,
                        split=args.split,
                        data_root=args.data_root,
                        sampler='random',
                        batch_images=args.batch_images,
                        meta=dataset_specs,
                        rgb_mean=mean_,
                        feat_stride=margs.feat_stride,
                        label_stride=margs.feat_stride,
                        origin_size=args.origin_size,
                        crop_size=args.crop_size,
                        scale_rate_range=[float(_) for _ in args.scale_rate_range.split(',')],
                        transformer=None,
                        transformer_image=ts.Compose(_get_transformer_image()),
                        prefetch_threads=args.prefetch_threads,
                        prefetcher_type=args.prefetcher,)
    dataiter.reset()
    # optimizer
    assert args.to_epoch is not None
    if args.stop_epoch is not None:
        assert args.stop_epoch > args.from_epoch and args.stop_epoch <= args.to_epoch
    else:
        args.stop_epoch = args.to_epoch
    from_iter = args.from_epoch * dataiter.batches_per_epoch
    to_iter = args.to_epoch * dataiter.batches_per_epoch
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
        scheduler = LinearScheduler(updates=to_iter+1, frequency=50,
                                    stop_lr=min(base_lr/100., 1e-6),
                                    offset=from_iter)
    elif lr_params['type'] == 'poly':
        scheduler = PolyScheduler(updates=to_iter+1, frequency=50,
                                    stop_lr=min(base_lr/100., 1e-8),
                                    power=0.9,
                                    offset=from_iter)
    optimizer_params = {
        'learning_rate': base_lr,
        'momentum': 0.9,
        'wd': args.weight_decay,
        'lr_scheduler': scheduler,
        'rescale_grad': 1.0/len(args.gpus.split(',')),
    }
    # initializer
    net_args = None
    net_auxs = None
    if args.weights is not None:
        net_args, net_auxs = mxutil.load_params_from_file(args.weights)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
    #
    to_model = osp.join(args.output, '{}_ep'.format(args.model))
    mod = _get_module(args, margs, dargs)
    mod.fit(
        dataiter,
        eval_metric=_get_metric(),
        # batch_end_callback=mx.callback.Speedometer(dataiter.batch_size, 1),
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
        
        conf = [len(class_flags[j].intersection(pred_flags[k])) for j in xrange(num_classes) for k in xrange(num_classes)]
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
                speed = 1.*self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i+1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}pixel acc: {:.2f}%, mean acc: {:.2f}%, mean iou: {:.2f}%'.\
                format(name, acc*100, cls_accs.mean()*100, ious.mean()*100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(cls_accs*100))
                logger.info('\n{}'.format(ious*100))
        
        return acc, cls_accs, ious
    
    def overall_scores(self, logger=None):
        acc, cls_accs, ious = self.scores(None, logger)
        return acc, cls_accs.mean(), ious.mean()

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
    else:
        raise NotImplementedError('Unknown phase: {}'.format(args.phase))

