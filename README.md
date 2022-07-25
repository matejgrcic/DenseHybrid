# DenseHybrid
Official implementation of ECCV2022 paper **DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition** [[arXiv]](https://arxiv.org/pdf/2207.02606.pdf)

**Info:** Upload in progress..

### Abstract
Anomaly detection can be conceived either through generative modelling of regular training data or by discriminating with respect to negative training data. These two approaches exhibit different failure modes. Consequently, hybrid algorithms present an attractive research goal. Unfortunately, dense anomaly detection requires translational equivariance and very large input resolutions. These requirements disqualify all previous hybrid approaches to the best of our knowledge. We therefore design a novel hybrid algorithm based on reinterpreting discriminative logits as a logarithm of the unnormalized joint distribution p*(x,y). Our model builds on a shared convolutional representation from which we recover three dense predictions: i) the closed-set class posterior P(y|x), ii) the dataset posterior P(din|x), iii) unnormalized data likelihood p*(x). The latter two predictions are trained both on the standard training data and on a generic negative dataset. We blend these two predictions into a hybrid anomaly score which allows dense open-set recognition on large natural images. We carefully design a custom loss for the data likelihood in order to avoid backpropagation through the untractable normalizing constant Z(θ). Experiments evaluate our contributions on standard dense anomaly detection benchmarks as well as in terms of open-mIoU - a novel metric for dense open-set performance. Our submissions achieve state-of-the-art performance despite neglectable computational overhead over the standard semantic segmentation baseline.

## Datasets
Cityscapes can be downloaded from [here](https://www.cityscapes-dataset.com/).

StreetHazards can be downloaded from [here](https://github.com/hendrycks/anomaly-seg).

Fishyscapes validation subsets with the appropriate structure: [FS LAF](https://drive.google.com/file/d/1fwl8jn4NLAp0LShOEZHYNS4CKdyEAt4L/view?usp=sharing), [FS Static](https://drive.google.com/file/d/1iWuoA218HweS9uuaPZvD5SJ-R93cTBHo/view?usp=sharing).


## Evaluation

### Weights

DeepLabV3+ trained on Cityscapes by NVIDIA: [weights](https://drive.google.com/file/d/1CKB7gpcPLgDLA7LuFJc46rYcNzF3aWzH/view?usp=sharing) 

Fine-tuned DeepLabV3+ (for Fishyscapes): [weights](https://drive.google.com/file/d/1MZhINlNrXQlEyByUxypBebZQECqWAhlL/view?usp=sharing) 

Trained LDN-121 semseg model on StreetHazards: [weights](https://drive.google.com/file/d/1Mf1sNVUhTtT1XexO-afco9577hzV5_kQ/view?usp=sharing) 

Fine-tuned LDN-121 semseg model on StreetHazards: [weights](https://drive.google.com/file/d/1vDXp-rySo-ASRh71O4h_MNiv_f-gFDm1/view?usp=sharing) 

### Dense anomaly detection

Fishyscapes LostAndFound val results:
> python evaluate_ood.py --dataroot LF_DATAROOT --dataset lf --folder OUTPUT_DIR --params WEIGHTS_FILE

Fishyscapes Static val results:
> python evaluate_ood.py --dataroot STATIC_DATAROOT --dataset static --folder OUTPUT_DIR --params WEIGHTS_FILE

StreetHazards results:
> python evaluate_ood.py --dataroot SH_DATAROOT --dataset street-hazards --folder OUTPUT_DIR --params WEIGHTS_FILE

### Dense open-set recognition

StreetHazards:
> python evaluate_osr.py --dataroot SH_DATAROOT --model WEIGHTS_FILE

## Training

Fine-tune DeepLabV3+ on Cityscapes with negatives:
> python dlv3_cityscapes.py --dataroot CITY_DATAROOT --neg_dataroot ADE_DATAROOT --exp_name EXP_NAME

Train LDN-121 on StreetHazards:
> python ldn_semseg.py --dataroot SH_DATAROOT --exp_name EXP_NAME

Fine-tune LDN-121 on StreetHazards with negatives:
> python ldn_streethazards.py --dataroot SH_DATAROOT --neg_dataroot ADE_DATAROOT --exp_name EXP_NAME --model MODEL_INIT


