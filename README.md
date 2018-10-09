## class-balanced self-training
MXNet implementation of our method for adapting semantic segmentation from the synthetic dataset (source domain) to the real dataset (target domain). Based on this implementation, our result is ranked 3rd in the VisDA Challenge.

Contact: Yi-Hsuan Tsai (wasidennis at gmail dot com) and Wei-Chih Hung (whung8 at ucmerced dot edu)

Requirements:
[MXNet 1.3.0](https://mxnet.apache.org/install/index.html?platform=Linux&language=Python&processor=GPU)
PIL
Python 2.7.x


Performance:
GTA2city:

	|Road|SW|Build|Wall|Fence|Pole|Traffic Light|Traffic Sign|Veg.|Terrain|Sky|Person|Rider|Car|Truck|Bus|Train|Motor|Bike|Mean
	------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------
	IoU|88.0|56.2|77.0|27.4|22.4|40.7|47.3|40.9|82.4|21.6|60.3|50.2|20.4|83.8|35.0|51.0|15.2|20.6|37.0|46.2

SYNTHIA2City:

# data
[GTA-5](https://download.visinf.tu-darmstadt.de/data/from_games/)
Since GTA-5 contains images with different resolutions, we recommend resize all images to 1052x1914. 

[Cityscapes](https://www.cityscapes-dataset.com/)

[SYNTHIA-RAND-CITYSCAPES](http://synthia-dataset.net/download/808/)

# source trained models 
(put source trained model in models/ folder)
[GTA-5](https://www.dropbox.com/s/idnnk398hf6u3x9/gta_rna-a1_cls19_s8_ep-0000.params?dl=0)
[SYNTHIA](https://www.dropbox.com/s/l6oxhxhovn2l38p/synthia_rna-a1_cls16_s8_ep-0000.params?dl=0)

# adapded models
[GTA2City](https://www.dropbox.com/s/1suk8xs48itd0fa/cityscapes_rna-a1_cls19_s8_ep-0000.params?dl=0)
[SYNTHIA2CIty]()

# spatial priors 
(put spatial prior in spatial_prior/gta/ folder)
[GTA-5](https://www.dropbox.com/s/o6xac8r3z30huxs/prior_array.mat?dl=0)

# self-training 
(export PYTHONPATH=PYTHONPATH:./)
we use a small class patch mining strategy to mine the patches including small classes. To turn off small class mining, set "--mine-port 0.0".  
GTA2Cityscapes:

CBST-SP:


CBST:

ST:

SYNTHIA2City:

CBST:

ST:

# evaluate
Cityscapes

GTA-5

SYNTHIA

This code heavily borrow [ResNet-38](https://github.com/itijyou/ademxapp)

Contact: yzou2@andrew.cmu.edu

If you finds this codes useful, please cite:

@InProceedings{Zou_2018_ECCV,
author = {Zou, Yang and Yu, Zhiding and Vijaya Kumar, B.V.K. and Wang, Jinsong},
title = {Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}

=======

Requirements:
MxNet 1.3.0
PIL
Python 2.7.x


Performance:
GTA2city:

SYNTHIA2City:

# data
[GTA-5](https://download.visinf.tu-darmstadt.de/data/from_games/)
Since GTA-5 contains images with different resolutions, we recommend resize all images to 1052x1914. 

[Cityscapes](https://www.cityscapes-dataset.com/)

[SYNTHIA-RAND-CITYSCAPES]()

# source trained models 
(put source trained model in models/ folder)
[GTA-5](https://www.dropbox.com/s/idnnk398hf6u3x9/gta_rna-a1_cls19_s8_ep-0000.params?dl=0)
[SYNTHIA]()

# adapded models
[GTA2City](https://www.dropbox.com/s/1suk8xs48itd0fa/cityscapes_rna-a1_cls19_s8_ep-0000.params?dl=0)
[SYNTHIA2CIty]()

# spatial priors 
(put spatial prior in spatial_prior/gta/ folder)
[GTA-5]()

# self-training 
(export PYTHONPATH=PYTHONPATH:./)
we use a small class patch mining strategy to mine the patches including small classes. To turn off small class mining, set "--mine-port 0.0".  
SYNTHIA2City:
CBST:
python issegm/script-self-paced-self-trained-segresnet.py --num-round 12 --test-scales 1850 --scale-rate-range 0.7,1.3 --dataset synthia --dataset-tgt cityscapes --split train --split-tgt val --data-root /home/datasets/RAND_CITYSCAPES --data-root-tgt /home/datasets/cityscapes/data_original --output spst_syn2city/cbst_eccv_min0-8 --model cityscapes_rna-a1_cls16_s8 --weights models/synthia_rna-a1_cls16_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size 1280 --origin-size-tgt 2048 --init-tgt-port 0.2 --init-src-port 0.02 --max-src-port 0.06 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/self-paced-self-trained-segresnet-public-v1.py --kc-policy cb --prefetch-threads 2 --gpus 2 --with-prior False

GTA2Cityscapes:
CBST-SP:
python issegm/script-self-paced-self-trained-segresnet.py --num-round 5 --test-scales 1850 --scale-rate-range 0.7,1.3 --dataset gta --dataset-tgt cityscapes --split train --split-tgt val --data-root DATA_ROOT_GTA5 --data-root-tgt DATA_ROOT_CITYSCAPES --output spst_gta2city/cbst-sp --model cityscapes_rna-a1_cls19_s8 --weights models/gta_rna-a1_cls19_s8_ep-0000.params --batch-images 2 --crop-size 500 --origin-size-tgt 2048 --init-tgt-port 0.15 --init-src-port 0.03 --seed-int 0 --mine-port 0.8 --mine-id-number 3 --mine-thresh 0.001 --base-lr 1e-4 --to-epoch 2 --source-sample-policy cumulative --self-training-script issegm/self-paced-self-trained-segresnet.py --kc-policy cb --prefetch-threads 2 --gpus 0 --with-prior True

For CBST, set "--with-prior False". For ST, set "--kc-policy global" and "--with-prior False".


# evaluate
Cityscapes

GTA-5

SYNTHIA

This code heavily borrow [ResNet-38](https://github.com/itijyou/ademxapp)

Contact: yzou2@andrew.cmu.edu

If you finds this codes useful, please cite:


> @InProceedings{Zou_2018_ECCV,
author = {Zou, Yang and Yu, Zhiding and Vijaya Kumar, B.V.K. and Wang, Jinsong},
title = {Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
