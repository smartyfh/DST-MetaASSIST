# DST-MetaASSIST

This is the implementation of our work: **MetaASSIST: Robust Dialogue State Tracking with Meta Learning. Fanghua Ye, Xi Wang, Jie Huang, Shenghui Li, Samuel Stern, Emine Yilmaz. EMNLP 2022.** [[paper](https://arxiv.org/abs/2210.12397)]

## Abstract

Existing dialogue datasets contain lots of noise in their state annotations. Such noise can hurt model training and ultimately lead to poor generalization performance. A general framework named ASSIST has recently been proposed to train robust dialogue state tracking (DST) models. It introduces an auxiliary model to generate pseudo labels for the noisy training set. These pseudo labels are combined with vanilla labels by a common fixed weighting parameter to train the primary DST model. Notwithstanding the improvements of ASSIST on DST, tuning the weighting parameter is challenging. Moreover, a single parameter shared by all slots and all instances may be suboptimal. To overcome these limitations, we propose a meta learning-based framework MetaASSIST to adaptively learn the weighting parameter. Specifically, we propose three schemes with varying degrees of flexibility, ranging from slot-wise to both slot-wise and instance-wise, to convert the weighting parameter into learnable functions. These functions are trained in a meta-learning manner by taking the validation set as meta data. Experimental results demonstrate that all three schemes can achieve competitive performance. Most impressively, we achieve a state-of-the-art joint goal accuracy of 80.10% on MultiWOZ 2.4.

## Usage

### I. Requirements
Install [PyTorch](https://pytorch.org/get-started/locally/) and [Transformers](https://huggingface.co/docs/transformers/installation)
```console
❱❱❱ pip install -r requirements.txt
```

And also install [higher](https://github.com/facebookresearch/higher) from source, a library providing support for higher-order optimization
```console
❱❱❱ git clone git@github.com:facebookresearch/higher.git
❱❱❱ cd higher
❱❱❱ pip install .
```

### II. Model Training
Please refer to each method for the details.

+ [AUX-DST](https://github.com/smartyfh/DST-MetaASSIST/tree/main/AUX-DST)
+ [STAR](https://github.com/smartyfh/DST-MetaASSIST/tree/main/STAR)
+ [SOM-DST](https://github.com/smartyfh/DST-MetaASSIST/tree/main/SOM-DST)

<font color='red'>Note: In this repo, we only show how to train the primary model using the new framework MetaASSIST. If you are also interested in training the auxiliary model from scratch, please refer to [this repo](https://github.com/smartyfh/DST-ASSIST) for details.</font>


## Citation

```bibtex
@inproceedings{ye2022metaassist,
  title={MetaASSIST: Robust Dialogue State Tracking with Meta Learning},
  author={Ye, Fanghua and Wang, Xi and Huang, Jie and Li, Shenghui and Stern, Samuel and Yilmaz, Emine},
  journal={arXiv preprint arXiv:2210.12397},
  year={2022}
  }
```
