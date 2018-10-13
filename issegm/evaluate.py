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

def parse_split_file(dataset, split, data_root=''):
    split_filename = 'issegm/data_list/{}/{}.lst'.format(dataset, split)
    image_list = []
    label_list = []
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
            label_list.append(os.path.join(data_root, fields[1]))
    return image_list, label_list

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

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

    model_specs = {
        # model
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
    parser.add_argument('--weights', default=None,
                        help='The path of a pretrained model.')
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
    parser.add_argument('--prefetch-threads', dest='prefetch_threads',
                        help='The number of threads to fetch data.',
                        default=1, type=int)
    parser.add_argument('--cache-images', dest='cache_images', 
                        help='If cache images, e.g., 0/1',
                        default=None, type=int)
    parser.add_argument('--log-file', dest='log_file',
                        default=None, type=str)
    parser.add_argument('--debug',
                        help='True means logging debug info.',
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

    if args.model is None:
        raise NotImplementedError('Missing argument: args.model')
    
    if args.log_file is None:
        if args.phase == 'val':
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
    cache_images = True
    cmap_path = 'data/shared/cmap.pkl'
    mx_workspace = 1650
    if dataset == 'cityscapes':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels import id2label, trainId2label
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
    elif dataset == 'cityscapes16':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels_cityscapes_synthia import id2label, trainId2label
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
        max_shape = np.array((1052, 1914))
        mx_workspace = 8192
    elif dataset == 'synthia':
        sys.path.insert(0, 'data/cityscapesscripts/helpers')
        from labels_synthia import id2label, trainId2label
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
        max_shape = np.array((760, 1280))
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
                net = fcrna_model_a1(margs.classes, margs.feat_stride, bootstrapping=True)
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

#@profile
def _val_impl(args, model_specs, logger):
    assert args.prefetch_threads == 1
    assert args.weights is not None
    
    margs = argparse.Namespace(**model_specs)
    dargs = argparse.Namespace(**get_dataset_specs(args, model_specs))

    image_list, label_list = parse_split_file(margs.dataset, args.split)
    net_args, net_auxs = mxutil.load_params_from_file(args.weights)
    net = None
    mod = _get_module(args, margs, dargs, net)
    has_gt = args.split in ('train', 'val',)
    
    crop_sizes = sorted([int(_) for _ in args.test_scales.split(',')])[::-1]
    crop_size = crop_sizes[0]
    # TODO: multi-scale testing
    assert len(crop_sizes) == 1, 'multi-scale testing not implemented'
    label_stride = margs.feat_stride

    save_dir_color = osp.join(args.output, 'color')
    save_dir_labelId = osp.join(args.output, 'labelId')
    save_dir_trainId = osp.join(args.output, 'trainId')
    _make_dirs(save_dir_color)
    _make_dirs(save_dir_labelId)
    _make_dirs(save_dir_trainId)
    
    x_num = len(image_list)
    
    do_forward = True
    # for all images that has the same resolution
    if do_forward:
        batch = None
        transformers = [ts.Scale(crop_size, Image.CUBIC, False)]
        transformers += _get_transformer_image()
        transformer = ts.Compose(transformers)
    
    scorer = ScoreUpdater(dargs.valid_labels, margs.classes, x_num, logger)
    scorer.reset()
    start = time.time()
    done_count = 0
    ############################
    for i in xrange(x_num):
        ############################ network forward on single image
        sample_name = osp.splitext(osp.basename(image_list[i]))[0]
        im_path = osp.join(args.data_root, image_list[i])
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
                dataiter = mx.io.NDArrayIter(input_data, input_label)
                batch = dataiter.next()

                mod.bind(dataiter.provide_data, dataiter.provide_label, for_training=False, force_rebind=True)
                # since we could use different transformers, but the same variables in the loop.
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
        # batch = None
        interp_preds = interp_preds_as(rim.shape[:2], net_preds, pred_stride, imh, imw)
        if args.save_results:
            # compute pixel-wise predictions
            pred_label = interp_preds.argmax(0)
            # with trainIDs
            pred_label_trainId = pred_label.copy()
            # with labelIDs
            if dargs.id_2_label is not None:
                pred_label = dargs.id_2_label[pred_label_trainId]
            # #
            # save predicted label with trainIDs, labelIDs, colors
            out_path_trainId = osp.join(save_dir_trainId, '{}.png'.format(sample_name))
            out_path_labelId = osp.join(save_dir_labelId, '{}.png'.format(sample_name))
            out_path_color = osp.join(save_dir_color, '{}.png'.format(sample_name))
            im_to_save_trainId = Image.fromarray(pred_label_trainId.astype(np.uint8))
            im_to_save = Image.fromarray(pred_label.astype(np.uint8))
            im_to_save_labelId = im_to_save.copy()
            # save color prediction
            im_to_save.putpalette(dargs.cmap.ravel())
            im_to_save.save(out_path_color)
            im_to_save_trainId.save(out_path_trainId)
            im_to_save_labelId.save(out_path_labelId)

        else:
            assert not has_gt
        
        done_count += 1
        if not has_gt:
            logger.info('Done {}/{} with speed: {:.2f}/s'.format(i+1, x_num, 1.*done_count / (time.time() - start)))
            continue
        if args.split in ('train', 'train+', 'val'):
            label_path = osp.join(args.data_root, label_list[i])
            label = np.array(Image.open(label_path), np.uint8)
        
        # save correctly labeled pixels into an image
            out_path = osp.join(save_dir_color, 'correct', '{}.png'.format(sample_name))
            _make_dirs(osp.dirname(out_path))
            invalid_mask = np.logical_not(np.in1d(label, dargs.valid_labels)).reshape(label.shape)
            Image.fromarray((invalid_mask*255 + (label == pred_label)*127).astype(np.uint8)).save(out_path)
        
            scorer.update(pred_label, label, i)

    logger.info('Done in %.2f s.', time.time() - start)

if __name__ == "__main__":
    util.cfg['choose_interpolation_method'] = True
    
    args, model_specs = parse_args()
    
    if len(args.output) > 0:
        _make_dirs(args.output)
    
    logger = util.set_logger(args.output, args.log_file, args.debug)
    logger.info('start with arguments %s', args)
    logger.info('and model specs %s', model_specs)

    if args.phase == 'val':
        _val_impl(args, model_specs, logger)
    else:
        raise NotImplementedError('Unknown phase: {}'.format(args.phase))

