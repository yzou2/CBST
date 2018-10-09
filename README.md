## class-balanced self-training
<<<<<<< HEAD

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
>>>>>>> 286bb776caa88e5ce2a68efa4983b63754a99b42
