from __future__ import print_function # Only Python 2.x
import subprocess
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description='Conservative self-trained segmentation ResNet-38.')
### self-trained network
parser.add_argument('--dataset', default=None,
                    help='The source dataset to use, e.g. cityscapes.')
parser.add_argument('--dataset-tgt', dest='dataset_tgt', default=None,
                    help='The target dataset to use, e.g. cityscapes.')
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
parser.add_argument('--weights', default=None,
                    help='The path of a pretrained model.')
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
parser.add_argument('--init-tgt-port', dest='init_tgt_port',
                    help='The initial portion of pixels selected in target dataset, both by global and class-wise threshold',
                    default=0.3, type=float)
parser.add_argument('--init-src-port', dest='init_src_port',
                    help='The initial portion of images selected in source dataset',
                    default=0.3, type=float)
parser.add_argument('--max-tgt-port', dest='max_tgt_port',
                    help='The max portion of pixels selected in target dataset, both by global and class-wise threshold',
                    default=0.5, type=float)
parser.add_argument('--max-src-port', dest='max_src_port',
                    help='The max portion of images selected in source dataset',
                    default=0.06, type=float)
parser.add_argument('--step-tgt-port', dest='step_tgt_port',
                    help='The step portion of pixels selected in target dataset, both by global and class-wise threshold',
                    default=0.05, type=float)
parser.add_argument('--step-src-port', dest='step_src_port',
                    help='The step portion of images selected in source dataset',
                    default=0.0025, type=float)
parser.add_argument('--epoch-per-round', dest='epr',
                    help='Training epochs in each round',
                    default=2, type=int)
parser.add_argument('--seed-int', dest='seed_int',
                    help='The random seed',
                    default=0, type=int)
parser.add_argument('--mine-port', dest='mine_port',
                    help='The portion of data being mined',
                    default=0.5, type=float)
parser.add_argument('--mine-id-number', dest='mine_id_number',
                    help='Thresholding value for deciding mine id',
                    default=3, type=int)
parser.add_argument('--mine-thresh', dest='mine_thresh',
                    help='The threshold to determine the mine id',
                    default=0.001, type=float)
parser.add_argument('--with-prior', dest='with_prior',
                    help='with prior',
                    default='False', type=str)
parser.add_argument('--num-round', dest='num_round',
                    help='the number of round in self-training',
                    default=12, type=int)
parser.add_argument('--test-scales', dest='test_scales',
                    help='Lengths of the longer side to resize an image into, e.g., 224,256.',
                    default=None, type=str)
parser.add_argument('--scale-rate-range', dest='scale_rate_range',
                    help='The range of rescaling',
                    default='0.7,1.3', type=str)
parser.add_argument('--base-lr', dest='base_lr',
                    help='The lr to start from.',
                    default=None, type=float)
parser.add_argument('--to-epoch', dest='to_epoch',
                    help='The number of epochs to run.',
                    default=None, type=int)
parser.add_argument('--source-sample-policy', dest='source_sample_policy',
                    help='The sampling policy in source domain, only support "cumulative" and "random" ',
                    default='cumulative', type=str)
parser.add_argument('--self-training-script', dest='self_training_script',
                    help='The path of conservative self-trained neural network .py file',
                    default=None, type=str)
parser.add_argument('--kc-policy', dest='kc_policy',
                    help='The kc determination policy, currently only "global" and "cb" (class-balanced)',
                    default=None, type=str)
# system
parser.add_argument('--prefetch-threads', dest='prefetch_threads',
                    help='The number of threads to fetch data.',
                    default=1, type=int)
parser.add_argument('--gpus', default='0',
                    help='The devices to use, e.g. 0,1,2,3', type=str)

def main():
    args = parser.parse_args()
    addr_weights = args.weights

    mine_address = args.output + '/stats'
    to_epoch = args.epr

    lr = args.base_lr
    tgt_port = args.init_tgt_port
    src_port = args.init_src_port

    for cround in range(args.num_round):
        if args.source_sample_policy == 'cumulative':
            rand_seed = args.seed_int
        elif args.source_sample_policy == 'random':
            rand_seed = cround
        else:
            raise NotImplementedError('Unknown source sample policy: {}'.format(args.source_sample_policy))

        # validation and generate pseudo-label map
        cmd_val = ['python', args.self_training_script, '--dataset', args.dataset,'--dataset-tgt', args.dataset_tgt,'--data-root',args.data_root,
                   '--data-root-tgt', args.data_root_tgt,'--split-tgt', args.split_tgt, '--output', args.output, '--model', args.model, '--phase', 'val',
                   '--weights', addr_weights,'--mine-id-number', str(args.mine_id_number), '--test-scales', str(args.test_scales), '--gpus',args.gpus,
                   '--init-tgt-port', str(tgt_port), '--idx-round',str(cround), '--no-cudnn', '--test-flipping','--with-prior',str(args.with_prior),
                   '--kc-policy',args.kc_policy]
        for path in execute(cmd_val):
            print(path, end="")

        # model retraining
        cmd_val = ['python',args.self_training_script,'--gpus',args.gpus,'--dataset',args.dataset,'--split',args.split,'--dataset-tgt', args.dataset_tgt,
                   '--split-tgt',args.split_tgt,'--data-root',args.data_root,'--data-root-tgt',args.data_root_tgt,'--output',args.output,
                   '--init-src-port', str(src_port),'--model', args.model,'--batch-images',str(args.batch_images),'--crop-size', str(args.crop_size),
                   '--scale-rate-range',args.scale_rate_range,'--weights', addr_weights,'--idx-round',str(cround),'--base-lr', str(lr),'--to-epoch', str(to_epoch),
                   '--prefetch-threads',str(args.prefetch_threads),'--prefetcher','process','--mine-id-address', mine_address,
                   '--mine-port', str(args.mine_port),'--mine-thresh', str(args.mine_thresh),'--cache-images','0','--backward-do-mirror',
                   '--origin-size', str(args.origin_size),'--origin-size-tgt', str(args.origin_size_tgt),'--seed-int', str(rand_seed), '--phase','train']
        # run codes
        for path in execute(cmd_val):
            print(path, end="")

        # update parameters for the following pseduo-label generation stage
        addr_weights = args.output + '/' + str(cround) + '/' + args.model + '_ep-%04d.params' % (to_epoch)
        to_epoch = to_epoch + args.epr
        lr = lr/math.sqrt(2)
        if tgt_port < args.max_tgt_port:
            tgt_port = tgt_port + args.step_tgt_port
        if src_port < args.max_src_port:
            src_port = src_port + args.step_src_port


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

if __name__ == '__main__':
    main()
