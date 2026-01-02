SPA: Shape-aware Prototype Acquisition for Cross-Domain Few-Shot Medical

Image Segmentation

## 🚩摘要
Few-Shot Medical Image Segmentation (FSMIS) has emerged as a vital solution to data scarcity, yet it suffers from severe generalization degradation when transferred to unseen domains with distinct imaging physics. Existing cross-domain approaches primarily focus on aligning low-level frequency or implicit feature statistics. However, they often overfit volatile texture patterns while neglecting the anatomical topological shape, which exhibits inherent and robust invariance across heterogeneous modalities. To this end, we propose a novel Shape-aware Prototype Acquisition (SPA) framework, which explicitly establishes anatomical shape information as a physical anchor to bridge the domain gap. Specifically, we first design a Spatial Attention Enhancement (SAE) module to suppress domain-specific texture noise, purifying the feature space for effective enhancement. Subsequently, a Shape-Guided Matching (SGM) module is introduced to disentangle domain-invariant structural representations from the entangled features via explicit anatomical shape constraints. Furthermore, to overcome the structural loss in standard pooling, we propose a Graph Prototype Generator (GPG) that constructs intra-class topological relationships via graph construction to generate robust prototypes. Extensive experiments on three public medical imaging datasets demonstrate that SPA significantly outperforms state-of-the-art methods.

## 🔍SPA的架构
![](https://cdn.nlark.com/yuque/0/2026/png/35053082/1767323030901-2968af09-1ee9-4779-acfe-60f5e1fefc5e.png)



## 🗝️快速开始
### 🔖依赖
请下载如下依赖：

```python
dcm2nii
json5==0.8.5
jupyter==1.0.0
nibabel==2.5.1
numpy==1.24.4
opencv_python==4.11.0.86
Pillow>=8.1.1
sacred==0.8.7
scikit_learn==1.3.2
scikit-image==0.18.3
SimpleITK==2.5.2
torch==2.4.1
torchvision==0.19.1
matplotlib==3.7.5
scipy==1.16.0
```

### 📋数据集和预处理
1. 下载数据集：
+ Abdomen MRI：[Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/)
+ Abdomen CT：[Multi-Atlas Abdomen Labeling Challenge](https://www.synapse.org/#!Synapse:syn3193805/wiki/218292)
+ <font style="color:rgb(31, 35, 40);">Cardiac LGE and b-SSFP</font>：[Multi-sequence Cardiac MRI Segmentation dataset](https://zmiclab.github.io/zxh/0/mscmrseg19/index.html)
+ <font style="color:rgb(31, 35, 40);">Prostate UCLH and NCI</font>：[Cross-institution Male Pelvic Structures](https://zenodo.org/records/7013610)
2. 数据预处理：
+ <font style="color:rgb(31, 35, 40);">Pre-processing is performed according to </font>[<font style="color:rgb(9, 105, 218);">Ouyang et al.</font>](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation/tree/2f2a22b74890cb9ad5e56ac234ea02b9f1c7a535)<font style="color:rgb(31, 35, 40);"> and we follow the procedure on their GitHub repository.</font>
3. 最终的数据存放在./data目录下，data目录结构如下：

```python
./data
├── ABD
│   ├── ABDOMEN_CT
│   │   ├── sabs_CT_normalized
│   │   └── supervoxels_5000
│   └── ABDOMEN_MR
│       ├── chaos_MR_T2_normalized
│       └── supervoxels_5000
├── Cardiac
│   ├── bSSFP
│   │   ├── cmr_bssFP_normalized
│   │   └── supervoxels_5000
│   ├── LGE
│   │   ├── cmr_LGE_normalized
│   │   └── supervoxels_5000
├── Prostate
│   ├── NCI
│   │   ├── tcia_p3t_normalized
│   │   └── supervoxels_......
│   └── UCLH
│       ├── biopsy_normalized
│       └── supervoxels_.......

```

### 📍下载预训练权重
| resnet50-imagenet | [https://download.pytorch.org/models/resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
| --- | --- |
| resnet50-coco | [https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) |
| resnet101-imagenet | [https://download.pytorch.org/models/resnet101-63fe2227.pth](https://download.pytorch.org/models/resnet101-63fe2227.pth) |
| resnet101-coco | [https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) |


1. 下载[resnet50-coco](https://download.pytorch.org/models/resnet101-63fe2227.pth)作为我们的预训练模型权重。
2. 创建checkpoint目录，并将下载好的模型放在checkpoint目录下。目录结构如下所示：

```python
\checkpoint
└── deeplabv3_resnet50_coco-cd0a2569.pth
```

### 🔥训练与推理
有6个训练任务，分别是:

1. Abdomen CT (train)-> MR(inference)
2. Abdomen MR (train)-> CT(inference)
3. Cardiac LGE(train) -> bSSFP(infernce)
4. Cardiac bSSFP (train) -> LGE(inference)
5. Prostate NCI (train) -> UCLH(inference)
6. Prostate UCLH (train) -> NCI (inference)



每个任务所对应的训练和推理命令被记录在下表中：

| | 任务 | 训练指令 | 推理指令 |
| --- | --- | --- | --- |
| 1.  | CT-> MR | ./scripts/train_on_ABDOMEN_CT.sh | ./scripts/test_ABDOMEN_CT2MR.sh |
| 2.  | MR->CT | ./scripts/train_on_ABDOMEN_MR.sh | ./scripts/test_ABDOMEN_MR2CT.sh |
| 3.  | LGE -> bSSFP | ./scripts/train_on_Cardiac_LGE.sh | ./scripts/test_Cardiac_LGE2bssFP.sh |
| 4.  | bSSFP -> LGE | ./scripts/train_on_Cardiac_bSSFP.sh | ./scripts/test_Cardiac_bssFP2LGE.sh |
| 5.  | NCI -> UCLH | ./scripts/train_on_Prostate_NCI.sh | ./scripts/test_Prostate_NCI2UCLH.sh |
| 6.  | UCLH -> NCI | ./scripts/train_on_Prostate_UCLH.sh | ./scripts/test_Prostate_UCLH2NCI.sh |


以CT->MR为例子：

训练：

```python
./scripts/train_on_ABDOMEN_CT.sh #文件需要有执行权限
```

推理：

```python
./scripts/test_ABDOMEN_CT2MR.sh
```

## 🌹致谢
<font style="color:rgb(31, 35, 40);">Our code is built upon the works of </font>[<font style="color:rgb(9, 105, 218);">SSL-ALPNet</font>](https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation)<font style="color:rgb(31, 35, 40);">, </font>[<font style="color:rgb(9, 105, 218);">ADNet</font>](https://github.com/sha168/ADNet)<font style="color:rgb(31, 35, 40);"> and </font>[<font style="color:rgb(9, 105, 218);">QNet</font>](https://github.com/ZJLAB-AMMI/Q-Net)<font style="color:rgb(31, 35, 40);">, we appreciate the authors for their excellent contributions!</font>

