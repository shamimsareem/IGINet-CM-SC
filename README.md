# Introduction
This repository is for **IGINet: Integrating Geometric Information to
Enhance inter-modal interaction for Fine-Grained
Image Captioning**. We have integrate geometry feature with the original X-Linear Network for image captioning (https://github.com/JDAI-CV/image-captioning). Our paper can be found https://link.springer.com/article/10.1007/s00530-024-01608-1. You can send email: shamim2@mail.ustc.edu.cn for any type of difficulty. 


For **CM-SC: Cross-modal spatial-channel attention network for image captioning**. You can send email: shamim2@mail.ustc.edu.cn for any type of difficulty. We have modified the original X-Linear attention by Cross-variance instead of bilinear pooling for both LSTM and Transformer for image captioning. Only change the low_rank.py (Here I put as Low_rank_CM_SC.py this is only for CM-SC attention, you have to run original X-Linear project and change low_rank.py by code inside Low_rank_CM_SC.py) file of Xlinear code(https://github.com/JDAI-CV/image-captioning). Our paper can be found (https://www.sciencedirect.com/science/article/pii/S0141938224003056).

Please cite with the following BibTeX:

```
@article{hossain2024iginet,
  title={IGINet: integrating geometric information to enhance inter-modal interaction for fine-grained image captioning},
  author={Hossain, MS and Aktar, S and Liu, W and Gu, N and Huang, Z},
  journal={Multimedia Systems},
  pages={1--13},
  year={2024},
  publisher={Springer}
}

@article{hossain2024cm,
  title={CM-SC: Cross-modal spatial-channel attention network for image captioning},
  author={Hossain, Md Shamim and Aktar, Shamima and Gu, Naijie and Huang, Zhangjin},
  journal={Displays},
  pages={102941},
  year={2024},
  publisher={Elsevier}
}
```

<p align="center">
  <img src="images/framework.jpg" width="800"/>
</p>


## Requirements
* Python 3
* CUDA 10
* numpy
* tqdm
* easydict
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)
* [coco-caption](https://github.com/ruotianluo/coco-caption)

## Data preparation
1. Download the [bottom up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to npz files
```
python2 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_10_100
```

2. Download the [annotations](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS) into the mscoco folder. More details about data preparation can be referred to [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)

3. Download [coco-caption](https://github.com/ruotianluo/coco-caption) and setup the path of __C.INFERENCE.COCO_PATH in lib/config.py

4. The pretrained models and results can be downloaded [here](https://drive.google.com/open?id=1a7aINHtpQbIw5JbAc4yvC7I1V-tQSdzb).

5. The pretrained SENet-154 model can be downloaded [here](https://drive.google.com/file/d/1CrWJcdKLPmFYVdVNcQLviwKGtAREjarR/view?usp=sharing).
6. Geometric features used for this work can be downloaded [GF-FRCNN MSCOCO](https://data.mendeley.com/preview/sf238jg557).

## Training
### Train IGINet model
```
bash experiments/xlan/train.sh
```

### Train IGINet model using self critical
Copy the pretrained model into experiments/xlan_rl/snapshot and run the script
```
bash experiments/xlan_rl/train.sh
```

### Train IGINet transformer model
```
bash experiments/xtransformer/train.sh
```

### Train IGINet transformer model using self critical
Copy the pretrained model into experiments/xtransformer_rl/snapshot and run the script
```
bash experiments/xtransformer_rl/train.sh
```

## Evaluation
```
CUDA_VISIBLE_DEVICES=0 python3 main_test.py --folder experiments/model_folder --resume model_epoch
```

## Acknowledgements
Thanks the contribution of [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch) and awesome PyTorch team.
