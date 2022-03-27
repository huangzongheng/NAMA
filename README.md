# deep metric learning

## requeirements
    OS: Linux 
    Memory: >=16GB 
    GPU Memory: >= 11G 

1. Install dependencies:
    - [pytorch>=1.6.1](https://pytorch.org/)
    - torchvision
    - [ignite=0.4.1](https://github.com/pytorch/ignite) 
    - [yacs](https://github.com/rbgirshick/yacs)
    - [apex](https://github.com/NVIDIA/apex)

2. Prepare dataset

    Create a directory to store reid datasets under this repo or outside this repo. Remember to set your path to the root of the dataset in `config/defaults.py` for all training and testing or set in every single config file in `configs/` or set in every single command.

    You can create a directory to store reid datasets under this repo via

    ```bash
    cd deep-metric-learning
    unzip ./datasets/DukeMTMC-reID.zip -d ./datasets
    unzip ./datasets/Market-1501-v15.09.15.zip -d ./datasets
    ```
    **data structure**
    （1）Market1501
    ```bash
    datasets
        market1501 # this folder contains 6 files.
            bounding_box_test/
            bounding_box_train/
            ......
    ```
    （2）DukeMTMC-reID
    ```bash
    datasets
        dukemtmc-reid
        	DukeMTMC-reID # this folder contains 8 files.
            	bounding_box_test/
            	bounding_box_train/
            	......
    ```


## Train
You can run these commands in  `.sh ` files for training different datasets of differernt loss.  You can also directly run code `sh *.sh` to run our demo after your custom modification.

1. Market1501, cross entropy loss + triplet loss(uncertainty)

```bash

{
LOSS=uc-tri-n # pairwise loss : cosface tri-n
LOSS_CLS=softmax # classification loss 
EPOCHS=240 # 120
S=32  # pairwise loss Scale
MARGIN=0.2  # 0.15
SEED=42
NECK=no # neck type： bnneck 
NAME=resnet50 # backbone arch
ARCH='uc-bn-no' # network arch
FBASE=nn_128 # MAM function type
for DATASET in market1501 dukemtmc  msmt17 #
do
  python3 train.py --config_file='configs/softmax_triplet.yml' MODEL.NAME $NAME  \
  MODEL.NECK $NECK  SOLVER.WEIGHT_DECAY_POLY 1e-1 DATASETS.COMBINE_ALL "False" \
  DATASETS.NAMES "$DATASET" SOLVER.IMS_PER_BATCH "(120)" SOLVER.WARMUP_ITERS "$(($EPOCHS / 12))" MODEL.ARCH $ARCH \
  SOLVER.MARGIN $MARGIN SOLVER.TRI_LOSS_WEIGHT 1 \
  SOLVER.WEIGHT_DECAY_NECK 5e-4 SOLVER.WEIGHT_DECAY 5e-4 \
  DATALOADER.NUM_INSTANCE "(10)"  SOLVER.CHECKPOINT_PERIOD "10" SOLVER.STEPS "$(($EPOCHS/4)), $(($EPOCHS*2/4))" \
  SOLVER.MAX_EPOCHS "$(($EPOCHS*3/4))" SOLVER.EVAL_PERIOD "$(($EPOCHS / 4))" SOLVER.S $S \
  MODEL.METRIC_LOSS_TYPE "$LOSS" MODEL.CLS_LOSS_TYPE $LOSS_CLS SEED $SEED MODEL.F_BASE $FBASE \
  OUTPUT_DIR "./logs/cls+tri/$DATASET/Exp-$ARCH-$FBASE-d$P_DECAY-$LOSS-s$S-$LOSS_CLS-10*12-m$MARGIN-g$G" \

  
done
}


```

2. run train and test on market and duke, using only classification loss
```bash
# train on market and duke 参数介绍同上
    
{
EPOCHS=240 # 120
SEED=42
S=16  # classification loss Scale
NECK=bnneck # bnneck
MARGIN=0.0
LOSS_CLS=uc-cls #arcface #
CTYPE=no
DATASET=market1501
NECK_DECAY=0.0
for DATASET in market1501 # dukemtmc  msmt17 #
do
      {
      if [ $DATASET = msmt17 ]
      then
        EPOCHS=$(($EPOCHS / 2))
      fi
      python3 train.py --config_file='configs/softmax_triplet.yml' MODEL.NAME "resnet50" \
      SOLVER.MARGIN $MARGIN SOLVER.WEIGHT_DECAY_POLY 1e-2 \
      SOLVER.WEIGHT_DECAY_NECK $NECK_DECAY  SOLVER.NORM_LU "[10, 110]" \
      DATASETS.NAMES "$DATASET" SOLVER.IMS_PER_BATCH "(120)" SOLVER.WARMUP_ITERS "$(($EPOCHS / 12))" MODEL.ARCH "clsuc" \
      DATALOADER.NUM_INSTANCE "(10)"  SOLVER.CHECKPOINT_PERIOD "10" SOLVER.STEPS "$(($EPOCHS/4)), $(($EPOCHS*2/4))" \
      SOLVER.MAX_EPOCHS "$(($EPOCHS*3/4))"  SOLVER.EVAL_PERIOD "$(($EPOCHS/12))" SOLVER.CS $S MODEL.NECK $NECK \
      MODEL.METRIC_LOSS_TYPE "none" MODEL.CLS_LOSS_TYPE $LOSS_CLS SEED $SEED MODEL.F_BASE $FBASE \
      OUTPUT_DIR "./logs/cls+/$DATASET/Exp-log-$FBASE-d$P_DECAY-$NECK-$LOSS_CLS-10*12-s$S-m$MARGIN-g$G"
      }
done
}
```


## Test
You can test your model's performance directly by running these commands in `.sh ` files after your custom modification. You can also change the configuration to determine which feature of BNNeck is used and whether the feature is normalized (equivalent to use Cosine distance or Euclidean distance) for testing.

Please replace the data path of the model and set the `PRETRAIN_CHOICE` as 'self' to avoid time consuming on loading ImageNet pretrained model.

1. test model and visiualize result

```bash

K=0.0   # 测试期间距离计算时加入不确定度的强度，一般用0
# TEST.VISRANK： 测试时可视化排序结果数量
for DATASET in dukemtmc # msmt17  # market1501
do
    {
      CUDA_VISIBLE_DEVICES=0 \
      python test.py --config_file='configs/softmax_triplet.yml' DATASETS.NAMES $DATASET \
      TEST.IMS_PER_BATCH 8 TEST.VISRANK 10 TEST.NECK_FEAT 'after' MODEL.ARCH "clsuc" MODEL.NAME 'resnet50' \
      TEST.WEIGHT "./logs/path to checkpoint *.pt" \
      OUTPUT_DIR "./logs/test/"
    }
done
```
