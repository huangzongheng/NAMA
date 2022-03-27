# partial reid test result

## train on market1501, test on partial-reid, partial-ilids

baseline: resize input img to 256X128 

exp1: keep the ratio, and resize img (crop) 

exp2: keep the ratio, and resize img then pad img to 256*128 (pad) 

exp3: resize input img width 1.5x, then use exp1 setting 

exp4: resize input img width 1.5x, then use exp2 setting 

| exp         | R1-reid  | R5   | R1-iLids | R5   |
| ----------- | -------- | ---- | -------- | ---- |
| VPM         | **67.7** | 81.9 | **65.5** | 74.8 |
| baseline    | 43.7     | 59.3 | 36.1     | 62.2 |
| exp1 crop   | 50.7     | 74.0 | 31.1     | 63.9 |
| exp2 pad    | 47.0     | 66.3 | 35.3     | 59.7 |
| exp3 c 1.5x | **65.0** | 81.7 | 33.6     | 63.0 |
| exp4 p 1.5x | 53.7     | 77.7 | 36.1     | 67.2 |
|             |          |      |          |      |

**在market上训练时采用的是256*128的输入，而partial-reid中完整图片比例接近3：1，partial-iLids完整图片比例接近2：1，因此将把图片宽度x1.5后，在partial-reid上提升显著**

## change stronger baseline
 
-bias：backbone的zero padding换成padding网络的bias。
-pad：训练时随机遮挡下半身一部分
+crop：测试时保持partial图片比例
+pad：测试时保持partial图片比例，并将图片pad到128x256

| exp                    | R1-reid  | R5   | R1-iLids | R5   |
| ---------------------- | -------- | ---- | -------- | ---- |
| VPM                    | **67.7** | 81.9 | **65.5** | 74.8 |
| baseline+crop          | **74.7** | 87.0 |          |      |
| baseline+pad           | 71.3     | 85.7 |          |      |
| baseline-bias+crop     | 74.0     | 86.7 |          |      |
| baseline-bias+pad      | 72.7     | 85.7 |          |      |
| baseline-pad+crop      | 67.3     | 79.7 |          |      |
| baseline-pad+pad       | **77.7** | 88.7 |          |      |
| baseline-bias-pad+crop | 68.0     | 82.7 |          |      |
| baseline-bias-pad+pad  | 75.0     | 87.0 |          |      |
| cosface+crop           | 72.0     | 83.7 |          |      |
| cosface+pad            | 71.3     | 85.0 |          |      |
| cosface-pad+crop       | 68.3     | 86.0 |          |      |
| cosface-pad+pad        | 66.7     | 81.7 |          |      |

## change padding mode

在网络输入img同时输入mask，在每层padding处，先将mask为0的pixel置0,。
-pad：训练时随机遮挡
+pad：测试时保持partial图片比例，并将图片pad到128x256
+mask：测试时只给图片加可见性mask，周围的pad不加mask
| exp               | R1-reid  | R5   | R1-iLids | R5   |
| ----------------- | -------- | ---- | -------- | ---- |
| VPM               | **67.7** | 81.9 | **65.5** | 74.8 |
| baseline+mask     | 6.3      | 18.0 |          |      |
| baseline+pad      | 73.3     | 86.0 |          |      |
| baseline-pad+mask | 74.3     | 88.3 |          |      |
| baseline-pad+pad  | 76.0     | 86.3 |          |      |

## 使用可变大小卷积核（对感受野做仿射变换）

训练时将垂直方向的mask变成resize
-pad, -mask均保持原图比例，没有将宽度扩充为1.4倍

ftaug: 训练时对输入图片随机crop一部分后拉伸
| exp                   | R1-reid  | R5   | R1-iLids | R5   |
| --------------------- | -------- | ---- | -------- | ---- |
| VPM                   | **67.7** | 81.9 | **65.5** | 74.8 |
| baseline-resize       | 54.7     | 66.7 |          |      |
| ft-affine-crop        | 58.7     | 76.7 |          |      |
| ft-affine-pad         | 60.0     | 78.3 |          |      |
| ft-affine-resize      | 58.7     | 69.3 |          |      |
| ft-all-crop           | 65.7     | 79.7 |          |      |
| ft-all-pad            | 63.7     | 81.7 |          |      |
| ft-all-resize         | 73.0     | 85.0 | 56.3     | 73.1 |
| baseline-ftaug-resize | 72.7     | 82.0 |          |      |
|                       |          |      |          |      |
感觉不一定是网络结构改变的影响，更有可能是网络从aug中适应了仿射变换的输入

## 使用子网络学习仿射参数，并对主网络中的卷积核做仿射变换

| exp                                                     | R1-reid      | R5   | R1-iLids | R5   | market-head mAP  | R1   |
| ------------------------------------------------------- | ------------ | ---- | -------- | ---- | ---------------- | ---- |
| VPM                                                     | **67.7**     | 81.9 | **65.5** | 74.8 |                  |      |
| 全图训练，1.4*w crop测试                                | **74.7**     | 87.0 | 51.3     | 75.6 |                  |      |
| pad训练，1.4*w pad测试                                  | **77.7**     | 88.7 | **63.9** | 84.9 |                  |      |
| 全图训练，crop测试                                      | 66.3         | 80.0 | 42.0     | 68.9 |                  |      |
| pad训练，pad测试                                        | 65.3         | 80.7 | 54.6     | 72.3 |                  |      |
| resize训练，resize测试                                  | 72.7         | 82.0 |          |      |                  |      |
| affine+resize+ft，resize测试                            | **75.3**     | 84.7 | 44.5     | 73.1 |                  |      |
| affine+resize-4*16，resize测试                          | 74.0         | 86.7 | 52.1     | 72.3 |                  |      |
| affine+resize头部完整-4*12，resize测试                  | **77.7**     | 86.0 | 57.1     | 78.2 |                  |      |
| affine+resize头部完整-10*16，resize测试                 | **78.7**     | 89.0 | **62.2** | 81.5 | 43.7             | 76.9 |
| affine+resize腿部完整-4*12，resize测试                  | 66.3         | 70.0 | 40.3     | 60.5 | 37.1             | 69.8 |
| affine+resize腿部完整-10*16，resize测试                 | 65.7         | 78.7 | 51.3     | 76.5 | 38.2             | 70.7 |
| baseline+resize头部大部分完整-4*16，resize测试          | 78.0         | 87.0 | 57.1     | 74.1 | 42.8             | 74.1 |
| affine+resize头部大部分完整-4*12，resize测试            | 73.0         | 85.0 | 48.7     | 68.1 | 44.8             | 76.0 |
| affine+resize头部大部分完整-10*16，resize测试           | **79.7**     | 88.7 | 58.0     | 78.2 | **47.6**         | 78.4 |
| affine+resize头部大部分完整+相对ratio-10*16，resize测试 | 80.0         | 89.3 | 52.9     | 75.6 | 46.8             | 79.4 |
| 增加ratio网络深度+相对ratio-10*16，resize测试           | 76.7         | 88.7 | 59.7     | 73.1 | 39.3             | 72.8 |
| ratio使用affine conv+cycle 一致性4*16                   | 76.0         | 87.0 | 58.0     | 79.0 | 34.4             | 69.6 |
| ratio使用affine conv+cycle 一致性4*4                    | 78.7         | 88.0 | 61.3     | 81.5 | 43.0             | 75.6 |
| 主干不加aug，只给ratio网络加aug，+cycle一致性4*16       | 72.0         | 81.3 | 52.1     | 73.1 | 24.1/86.2(full)  | 60.9 |
| 主干不加aug，只给ratio网络加aug，+cycle+it2一致性4*16   | 76.7         | 86.3 | 52.9     | 80.7 | 31.3/85.6(full)  | 71.9 |
| 主干不加aug，只给ratio网络加aug，+cycle+it2一致性4*16   | 67.1 (group) | 84.6 |          |      |                  |      |
| baseline不加aug                                         | 55.3         | 68.3 | 41.2     | 62.2 | 22.0 /86.5(full) | 56.1 |
| 主干不加aug，只给ratio网络加aug，+cycle+it3一致性4*16   | 76.0         | 88.3 | 51.3     | 73.1 | 13.1             | 39.8 | 增强了aug的幅度，反射pad
|                                                         |              |      |          |      |                  |      |
 

fix backbone, 只训练affine分支, 输入+aug
id+af:同时使用idloss和仿射参数自监督
af:只使用自监督
id+cycle： wo-augid loss+自监督加一致性约束,主干输入不加aug  
id+cycle-it2 ：wo-aug idloss+自监督加一致性约束，ratio迭代两次,主干输入不加aug  

| exp             | R1-reid | R5   | R1-iLids | R5   | market-head mAP | R1   |
| --------------- | ------- | ---- | -------- | ---- | --------------- | ---- |
| baseline-resize | 51.0    | 64.0 | 37.0     | 65.5 | 22.0            | 56.1 |
| ft id+af-resize | 59.3    | 76.0 | 42.0     | 70.6 |                 |      |
| ft af-resize    | 60.7    | 72.3 | 47.1     | 73.9 |                 |      |
| ft id+cycle     | 66.3    | 79.7 | 42.9     | 67.2 | 23.1            | 57.3 |
|                 | 68.0    | 80.3 | 47.1     | 73.1 | 28.1            | 65.4 |
| id+cycle-it2    | 69.0    | 82.7 | 47.1     | 73.1 | 27.8            | 64.1 |
| id+cycle-it3    | 74.0    | 85.7 | 36.1     | 58.8 | 7.9             | 28.8 | 增强了aug的幅度，反射pad
|                 |         |      |          |      |                 |      |

### 随机垂直crop market 0~0.5的图片（头部，腿部都可能crop）

| exp market1501-partial | map      | R1   |
| ---------------------- | -------- | ---- |
| baseline-ft resize     | **50.9** | 79.0 |
| affine-ft resize       | 50.0     | 77.0 |
| affine resize          | 47.0     | 75.1 |
|                        |          |      |

### 随机垂直crop market 0.15-0.45的图片（头部，腿部都可能crop）

| exp market1501-partial | map      | R1   |
| ---------------------- | -------- | ---- |
| baseline-ft resize     | 44.0     | 75.1 |
| affine-ft resize       | **45.6** | 76.9 |
| affine resize          | 42.8     | 74.0 |
|                        |          |      |

### 随机垂直crop market 0.15-0.45的图片（主要crop腿部）

| exp market1501-partial | map      | R1   |
| ---------------------- | -------- | ---- |
| baseline-ft resize     | **63.0** | 85.7 |
| affine-ft resize       | 62.3     | 85.5 |
| affine resize          | 61.7     | 84.8 |
|                        |          |      |

### 随机垂直crop market 0.25-0.55的图片（主要crop腿部）

| exp market1501-partial | map      | R1   |
| ---------------------- | -------- | ---- |
| baseline-ft resize     | **44.7** | 75.7 |
| affine-ft resize       | 42.3     | 73.7 |
| affine resize          | 43.4     | 75.0 |
|                        |          |      |

## 仿射预测分支实验

backbone：res18

| exp        | error  |     |
| ---------- | ------ | --- |
| cut layer1 | 0.0164 |     |
| cut layer2 | 0.0113 |     |
| cut layer3 | 0.0072 |     |
| cut layer4 | 0.0049 |     |


## 利用相对关系训练ratio预测器

在利用market自监督训练ratio预测器时，一般认为原图的ratio是1：1，在长宽方向分别缩放a:b后，图片的比例为a:b。训练时直接将ab作为gt训练。但是market中的比例不一定都是1：1，因此可以考虑缩放前后的相对关系，假设原图比例为h:w，则缩放后为ah:bw，同时我们认为market中图片的平均比例为1：1。因此可以设计下面两个损失函数：

Lr=(((裁剪预测ratio/裁剪ratio)/原图预测ratio)-1)**2
Lb=0.01 * (原图预测ratio)**2

其中Lr要求网络能预测相对关系，Lb则是一个很弱的约束，要求网络预测ratio的均值在1附近。要实现上述目标，我们需要Lr对ratio预测值的绝对关系完全不敏感，否则就会使得Lb失效。根据上述loss训练网络时，发现网络预测的ratio均值约为0.4，远小于1。

考虑到ratio的特性，它的对称点在1，如果两个变换满足a*b=1，那么他们是对称的，而nn的输出还是一个接近线性的关系，因此在计算loss和预测时应考虑这一点。
首先考虑loss，我们希望loss满足于ratio相似的对称性，即偏差的ratio=a和1/a时，loss应该相同。上述loss显然不满足。而如果将ratio的比例取对数，则可以满足对称性要求：

Lr=(log((裁剪预测ratio/裁剪ratio)/原图预测ratio))**2

经过这样的变换后，网络预测的ratio均值约为提升到0.93左右，已经比较接近1了。
下一步修改网络的输出层，原来网络直接采用一个fc预测ratio，即
ratio = (W*x+b)     # b初始化为1
若将上式修改为
ratio = exp(W*x+b)     # b初始化为0
则使ratio与x对称性相同了。
修改后，ratio均值提升到0.99左右，基本可以认为Lr对Lb没有影响。

**next：迭代求解ratio？**

## crop下半身market

baseline:全身图训练

baseline-affine-y2:加载baseline参数，手动设置affine conv y方向stride=2  

affine:用affine预测器预测affine conv参数

baseline-resize:训练输入图片用crop+resize作为aug

| exp market1501-partial | mAP y=1.0 | R1   | mAP y=0.5 | R1   |
| ---------------------- | --------- | ---- | --------- | ---- |
| vpm                    | 80.8      | 90.3 | 48.8      | 70.9 |
| baseline               | 87.7      | 95.0 | 19.1      | 38.9 |
| baseline-affine-y2     | 57.3      | 81.1 | 48.4      | 73.4 |
| affine                 | 85.6      | 94.0 | 43.8      | 72.1 |
| baseline-resize        |           |      |           |      |


### 改变输入图片尺寸
输入图片xy尺寸变化范围为（96~160）（192~320），但均为全身图，测试时尺寸为128*256
验证加入affine conv是否能帮助网络抵抗尺度变化
baseline:训练输入为128*256
normal：训练输入尺寸可变
affine：训练输入可变，同时ratio参数也随动变化
normal+：训练输入尺寸可变（64~192）（128~384）
affine+：训练输入可变，同时ratio参数也随动变化（64~192）（128~384）
normal++：训练输入尺寸可变（128~256）（256~512）
affine++：训练输入可变，同时ratio参数也随动变化（128~256）（256~512）
affine++ pretrain：训练输入可变，同时ratio参数也随动变化（128~256）（256~512）,加载训练好的参数
affine fix1.5 pretrain ：训练输入固定384x192，同时ratio参数设置为1.5 
affine++ wo pool：将res50的maxpool换成一个3x3conv，训练输入可变，同时ratio参数也随动变化（128~256）（256~512）
normal++ wo pool：将res50的maxpool换成一个3x3conv，训练输入尺寸可变（128~256）（256~512）
affine++ all pool：将res50的downsample全部替换为maxPooling，训练输入可变，同时ratio参数也随动变化（128~256）（256~512）
normal++ all pool：将res50的downsample全部替换为maxPooling，训练输入尺寸可变（128~256）（256~512）
baseline resize x1.5：baseline参数，输入size扩大1.5倍 
baseline affine x1.5：baseline参数，输入size扩大1.5倍，卷积size也扩大1.5倍

| market                 | mAP  | R1   |
| ---------------------- | ---- | ---- |
| baseline               | 86.3 | 94.8 |
| normal                 | 84.6 | 93.4 |
| affine                 | 84.0 | 93.6 |
| normal+                | 80.2 | 91.2 |
| affine+                | 79.7 | 91.3 |
| normal++               | 83.6 | 93.6 |
| affine++               | 83.9 | 93.4 |
| affine++ pretrain      | 84.2 | 94.3 |
| affine fix1.5 pretrain | 83.4 | 92.8 |
| normal++ wo pool       | 82.2 | 93.2 |
| affine++ wo pool       | 77.6 | 91.4 |
| normal++ all pool      | 83.0 | 93.9 |
| affine++ all pool      | 75.9 | 90.4 |
| baseline resize*1.5    | 30.7 | 58.8 |
| baseline affine*1.5    | 83.6 | 93.4 |

实验表明，经过适当的训练，普通的cnn也能适应不同大小的输入，而可变size的卷积在应对不同大小输入时，仍会导致性能损失，且损失的性能甚至超过普通的cnn。
然而对于没有经过不同size输入训练的cnn，可变size卷积确实可以帮助网络直接适应不同size的输入，但是性能损失虽然不算很大，但是也不能忽略。
因此利用可变size卷积，来处理输入图片小范围的形变似乎并不合算，一方面会引入成倍的计算和显存开销，另一方面还需要额外的网络准确的预测卷积核size参数，效果还不如直接给网络不同大小的输入，有点得不偿失。

增大输入size，避免可变卷积dilation小于1可以降低性能损失，但是仍不比标准卷积有优势。去除pooling层对可变卷积的性能损害比普通卷积更明显，因此训练困难不是pooling导致的。

保持backbone参数不变，仅改变输入size和卷积核stride
| input size    | mAP  | R1   | input size | mAP  | R1   | input size | mAP  | R1   |
| ------------- | ---- | ---- | ---------- | ---- | ---- | ---------- | ---- | ---- |
| 256 * 64      | 60.1 | 79.7 | 128 * 128  | 36.3 | 60.7 | 128 * 64   | 10.3 | 24.3 |
| 256 * 96      | 84.6 | 93.6 | 192 * 128  | 82.5 | 92.6 | 192 * 96   | 76.2 | 89.2 |
| **256 * 128** | 87.8 | 95.1 | 256 * 128  | 87.8 | 95.1 | 256 * 128  | 87.8 | 95.1 |
| 256 * 160     | 87.4 | 94.9 | 384 * 128  | 87.3 | 94.7 | 320 * 160  | 86.5 | 94.2 |
| 256 * 192     | 87.4 | 95.0 | 448 * 128  | 87.8 | 94.9 | 384 * 192  | 86.7 | 94.4 |
| 256 * 224     | 87.8 | 95.0 | 512 * 128  | 88.1 | 95.1 | 448 * 224  | 87.7 | 95.1 |
| 256 * 256     | 88.0 | 95.2 |            |      |      | 512 * 256  | 88.2 | 95.1 |

保持backbone参数不变，仅改变输入size
| input size | mAP  | R1   | normal++ | mAP  | R1   |
| ---------- | ---- | ---- | -------- | ---- | ---- |
| 256 * 128  | 87.9 | 95.0 |          | 83.6 | 93.6 |
| 192 * 96   | 82.0 | 92.5 |          |      |      |
| 128 * 64   | 24.6 | 50.2 |          |      |      |
| 320 * 160  | 80.3 | 90.9 |          | 85.6 | 93.6 |
| 384 * 192  | 54.9 | 77.0 |          | 81.3 | 91.7 |
| 448 * 224  | 25.4 | 49.3 |          | 68.7 | 84.6 |
| 512 * 256  | 11.1 | 27.0 |          | 48.0 | 70.0 |
更大的输入size不会影响可变卷积的性能，甚至还有提升，但更小的size对可变卷积性能有巨大的伤害

## vpm实验

normal:标准vpm
affine：使用可变卷积
wo aug：backbone输入不用垂直crop    
exp market-partial  
|       | mAP    | R1   | mAP    | R1   | mAP    | R1     | mAP    | R1     |
| ----- | ------ | ---- | ------ | ---- | ------ | ------ | ------ | ------ |
| ratio | normal | conv | affine | conv | normal | wo aug | affine | wo aug |
| 0.5   | 48.4   | 71.5 | 49.5   | 77.2 | 7.3    | 8.2    | 29.2   | 41.7   |
| 0.6   | 63.3   | 84.9 | 61.9   | 85.7 | 30.9   | 40.8   | 54.5   | 75.6   |
| 0.7   | 69.7   | 89.3 | 68.0   | 89.0 | 59.2   | 77.0   | 68.5   | 87.1   |
| 0.8   | 74.4   | 91.5 | 72.3   | 90.8 | 73.7   | 89.4   | 75.5   | 91.4   |
| 0.9   | 77.3   | 92.7 | 75.9   | 92.5 | 80.0   | 93.6   | 79.3   | 92.7   |
| 1.0   | 78.2   | 93.3 | 77.5   | 93.0 | 81.6   | 94.2   | 80.5   | 93.2   |
|       |        |      |        |      |        |        |        |        |

+warmup+label smooth+3*triplet weight
drop:计算part feature距离时随机drop一个分支

|       | mAP    | R1   | mAP    | R1   | mAP    | R1     | mAP    | R1     | mAP    | R1          | mAP    | R1       |
| ----- | ------ | ---- | ------ | ---- | ------ | ------ | ------ | ------ | ------ | ----------- | ------ | -------- |
| ratio | normal | conv | affine | conv | normal | wo aug | affine | wo aug | affine | wo aug+drop | affine | drop 0.5 |
| 0.5   | 48.3   | 72.4 | 46.3   | 71.6 | 7.8    | 8.7    | 31.2   | 47.6   | 29.6   | 43.2        | 32.7   | 49.9     |
| 0.6   | 62.1   | 84.6 | 58.7   | 82.8 | 30.8   | 41.2   | 52.6   | 73.1   | 52.6   | 73.0        | 53.3   | 74.1     |
| 0.7   | 68.4   | 88.2 | 64.1   | 86.0 | 61.4   | 78.7   | 66.7   | 84.2   | 67.7   | 85.1        | 67.0   | 84.9     |
| 0.8   | 73.9   | 90.8 | 71.8   | 90.1 | 74.3   | 89.6   | 74.9   | 90.8   | 75.5   | 91.5        | 75.8   | 91.5     |
| 0.9   | 78.3   | 92.5 | 76.5   | 92.0 | 79.5   | 93.0   | 78.6   | 92.7   | 78.8   | 92.7        | 79.9   | 93.0     |
| 1.0   | 80.8   | 93.5 | 78.4   | 92.8 | 81.0   | 93.9   | 79.9   | 93.1   | 80.2   | 93.1        | 81.6   | 93.8     |
|       |        |      |        |      |        |        |        |        |        |             |        |          |

训练时间从400 epoch减少到200 epoch
|       | mAP    | R1   | mAP    | R1   | mAP    | R1     | mAP    | R1     |
| ----- | ------ | ---- | ------ | ---- | ------ | ------ | ------ | ------ |
| ratio | normal | conv | affine | conv | normal | wo aug | affine | wo aug |
| 0.5   | 48.6   | 73.2 | 48.5   | 74.7 | 7.2    | 7.6    | 32.3   | 49.3   |
| 0.6   | 63.5   | 84.5 | 62.5   | 85.5 | 31.5   | 43.3   | 52.9   | 74.5   |
| 0.7   | 70.3   | 88.6 | 68.6   | 89.0 | 61.3   | 77.8   | 67.5   | 85.6   |
| 0.8   | 75.7   | 91.5 | 75.2   | 91.9 | 74.2   | 89.3   | 75.9   | 91.2   |
| 0.9   | 79.2   | 93.0 | 78.5   | 92.6 | 80.4   | 93.1   | 80.0   | 93.2   |
| 1.0   | 81.2   | 93.4 | 80.3   | 93.5 | 82.4   | 94.2   | 81.5   | 93.7   |
|       |        |      |        |      |        |        |        |        |

延伸测试范围
|       | mAP    | R1   | mAP    | R1   |
| ----- | ------ | ---- | ------ | ---- |
| ratio | normal | conv | affine | conv |
| 0.3   | 19.9   | 30.6 | 29.6   | 50.5 |
| 0.4   | 36.0   | 57.0 | 40.5   | 67.0 |
| 0.5   | 48.7   | 73.3 | 48.6   | 75.7 |
| 0.6   | 63.7   | 85.0 | 62.0   | 86.0 |
| 0.7   | 70.4   | 89.5 | 67.8   | 89.1 |
| 0.8   | 75.9   | 91.3 | 74.6   | 91.4 |
| 0.9   | 79.5   | 93.0 | 78.2   | 92.8 |
| 1.0   | 81.8   | 93.6 | 80.3   | 93.5 |
|       |        |      |        |      |

nparts=1
|       | mAP    | R1   | mAP    | R1   |
| ----- | ------ | ---- | ------ | ---- |
| ratio | normal | conv | affine | conv |
| 0.5   | 45.6   | 68.2 | 46.8   | 70.8 |
| 0.6   | 58.5   | 81.7 | 58.1   | 81.2 |
| 0.7   | 62.4   | 84.4 | 61.5   | 83.7 |
| 0.8   | 64.2   | 86.2 | 63.0   | 84.9 |
| 0.9   | 65.1   | 86.6 | 64.0   | 86.1 |
| 1.0   | 65.3   | 87.0 | 64.5   | 86.4 |
|       |        |      |        |      |

add relu
-p：使用预测的ratio
-full：sampler会利用一个完整batch的数据
-x:在x方向也加aug
drop:随机drop两个分支

|       | mAP      | R1   | mAP    | R1   | mAP      | R1   | mAP           | R1   | mAP        | R1   | mAP           | R1   |
| ----- | -------- | ---- | ------ | ---- | -------- | ---- | ------------- | ---- | ---------- | ---- | ------------- | ---- |
| ratio | aff-naug | conv | affine | conv | affine-p | conv | affine-p-full | conv | affine-p-x | conv | affine-p-drop | conv |
| 0.5   | 33.4     | 50.4 | 48.8   | 75.2 | 49.8     | 76.7 | 43.0          | 69.2 | 49.5       | 76.1 | 49.7          | 76.7 |
| 0.6   | 53.1     | 75.9 | 62.0   | 85.0 | 62.9     | 86.0 | 56.2          | 81.6 | 62.2       | 86.2 | 62.9          | 86.0 |
| 0.7   | 67.4     | 86.3 | 68.0   | 88.7 | 68.6     | 88.8 | 62.2          | 85.3 | 68.1       | 88.5 | 68.6          | 89.5 |
| 0.8   | 75.9     | 91.7 | 74.8   | 91.7 | 75.1     | 91.6 | 69.3          | 88.8 | 74.0       | 91.3 | 75.1          | 91.9 |
| 0.9   | 79.9     | 92.7 | 78.2   | 92.5 | 78.9     | 92.8 | 74.5          | 91.4 | 78.1       | 92.6 | 79.1          | 93.2 |
| 1.0   | 81.7     | 93.6 | 80.3   | 93.6 | 81.2     | 93.9 | 77.7          | 92.7 | 80.2       | 94.1 | 81.3          | 93.8 |
|       |          |      |        |      |          |      |               |      |            |      |               |      |

|               | R1     | R3   | R1       | R3   | R1            | R3   | R1         | R3   | R1            | R3   | R1  | R3  |
| ------------- | ------ | ---- | -------- | ---- | ------------- | ---- | ---------- | ---- | ------------- | ---- | --- | --- |
| dataset       | normal | conv | affine-p | conv | affine-p-full | conv | affine-p-x | conv | affine-p-drop | conv |     |     |
| partial-Reid  | 67.7   | 80.1 | 52.2     | 69.9 |               |      | 60.5       | 75.7 |               |      |     |     |
| partial-iLids | 58.0   | 71.4 |          |      |               |      |            |      |               |      |     |     |

# ema+triplet loss实验

结合byol思想，在reid模型中使用ema，计算triplet loss时仅收紧model和ema间正样本的类内距离，不强制推开类间距离（但是还是用了分类loss）
tri：标注triplet loss，但是detach dan
r-tri：相对triplet loss，detach dan
tri-n：标注triplet loss
+ema：计算距离矩阵时用model 特征和ema model特征两两计算距离，否则均使用model特征。
|                          | mAP  | R1   |
| ------------------------ | ---- | ---- |
| tri-0.1                  | 85.2 | 94.3 |
| tri-0.1 +ema0.9          | 86.1 | 94.8 |
| tri-n-0.1 +ema0.99       | 85.1 | 94.1 |
| r-tri-0.6                | 58.9 | 80.4 |
| r-tri-0.6 +ema0.9        | 86.5 | 94.7 |
| r-tri-0.6 +ema0.99       | 87.4 | 95.5 |
| r-tri-0.6 +ema0.999      | 87.5 | 95.3 |
| r-tri-0.6 +ema0.999 4*30 |      |      |
| r-tri-0.6 +ema0.999 4*64 | 80.9 | 92.2 | 85.5/94.0 max
| r-tri-0.6 +ema0.999 8*64 | 82.6 | 92.7 | 86.4/94.2 max
|                          |      |      |

# 行人属性+reid实验
测试：market / market to duke

baseline：单层fc作为分类器，训练使用tripletloss+softmax
baseline+att：加入属性标签，多任务训练
bottleneck：id分类器用fc+bn+fc
bottleneck+att：多任务训练,属性分类器用bottleneck
bottleneck+id+att：多任务训练，属性id分类器都用bottleneck

| market1501 / ->duke | mAP  | R1       | mAP  | R1   |
| ------------------- | ---- | -------- | ---- | ---- |
| baseline            | 82.4 | **94.1** | 23.6 | 40.6 |
| baseline+att        | 81.9 | 93.5     | 23.5 | 41.0 |
| bottleneck          | 82.5 | 93.5     | 20.8 | 36.5 |
| bottleneck+att      | 82.8 | 93.5     | 24.0 | 41.1 |
| bottleneck+id+att   | 82.7 | 93.3     | 23.3 | 40.6 |

# 行人属性+strong reid实验

baseline：id+triplet loss
+att：使用属性标签
+att+drop：属性分支embd层加dropout
+att+norm：属性分类器用余弦距离计算
+att+norm*2：属性分类器用2倍余弦距离计算
256d：输出维度用fc降维到256

| market1501           | mAP  | R1   |
| -------------------- | ---- | ---- |
| baseline             | 86.6 | 94.2 |
| baseline + att       | 86.7 | 94.7 |
| baseline +att+drop   | 85.9 | 94.1 |
| baseline +att+norm   | 86.8 | 94.6 |
| baseline +att+norm*2 | 87.0 | 94.2 |
| baseline 256d        | 83.8 | 93.4 |
| baseline 256d+drop   | 82.1 | 92.1 |
| baseline 1024d       | 84.5 | 93.9 |
|                      |      |      |

cf：加入attention局部分支，全局分支计算分类loss，局部分支+全局特征计算triplet loss
cf-trif：计算triplet loss，全局分支特征用于计算距离，但是不回传梯度。
mlp-n:属性分类器用n层mlp【512/256/128】

| duke                         | mAP  | R1   |
| ---------------------------- | ---- | ---- |
| baseline                     | 76.9 | 87.5 |
| baseline +att+norm*2         | 75.6 | 87.0 |
| baseline + att               | 72.9 | 85.4 |
| baseline 128d+att+norm*2     | 75.5 | 86.4 |
| baseline 128d+ att           | 75.9 | 86.5 |
| baseline 128d+ att-attention | 76.4 | 87.3 |
| baseline+cf                  | 77.1 | 86.9 |
| baseline+cf+trif             | 77.4 | 86.9 |
| baseline mlp4                | 76.4 | 87.2 |
| baseline mlp3                | 76.7 | 87.3 |
| baseline mlp2                | 74.9 | 86.2 |
| baseline mlp2 -128d          | 74.5 | 85.5 |
|                              |      |      |

# 使用属性refine ranking list

baseline：softmax+triplet，不使用属性

tri-attr：根据属性分类结果计算距离，权重可学（基函数为|x1-x2|,x1*x2,(x1-x2)^2）

tri-attr-label:只对属性和id标签均不同的样本计算属性距离。

tri-amargin：根据属性标签不同的数量来选择margin。

| market         | mAP  | R1   | duke     | mAP  | R1   |
| -------------- | ---- | ---- | -------- | ---- | ---- |
| baseline       | 86.6 | 94.2 | baseline | 76.9 | 87.5 |
| tri-attr       | 87.1 | 94.8 |          | 75.6 | 87.0 |
| tri-attr-label | 86.9 | 94.8 |          | 75.4 | 86.0 |
| tri-amargin    | 86.9 | 94.2 |          |      |      |


# 使用网络输出的一部分维度的feature

sel-1024d：从2048维输出中均匀选择出1024维特征
sel-1024d-train：从2048维输出中均匀选择出1024维特征，并用1024维特征进行训练
sel-1024d-tri：从2048维输出中均匀选择出1024维特征，并用1024维特征算triplet loss
mean-256d：相邻channel求均值得到256维特征

| market          | mAP  | R1   |
| --------------- | ---- | ---- |
| baseline        | 86.9 | 94.8 |
| sel-1024d       | 86.5 | 94.8 |
| sel-512d        | 85.4 | 94.4 |
| sel-256d        | 83.0 | 93.7 |
| mean-512d       | 85.3 | 94.4 |
| mean-256d       | 82.3 | 93.6 |
| sel-256d-train  | 80.2 | 92.8 |
| sel-1024d-train | 84.4 | 83.9 |
| sel-2048d-train | 85.8 | 94.4 |85.8/94.6
| sel-1024d-tri   |      |      |
| sel-256d-tri    | 81.8 | 93.2 |
|                 |      |      |

加一层fc来进行特征降维似乎还不如直接选一部分维度

# duke hard query

显著性坐标(h,w) label me
0034_c1_f0057636.jpg,  (4,4)
0061_c1_f0062281.jpg,  (3,4)
0043_c1_f0059677.jpg,  (4,3)
0046_c1_f0060037.jpg,  (4,6)
0053_c1_f0060659.jpg,  (5,6)
0140_c8_f0141968.jpg,  (3,3)
0167_c1_f0084519.jpg,  (5,3)
0174_c2_f0090920.jpg,  (5,3)
0183_c7_f0069034.jpg,  (4,4)
0188_c1_f0087974.jpg,  (3,4)
0244_c6_f0073324.jpg,  (2,4)
0247_c1_f0095658.jpg,  (3,4)
0272_c1_f0097338.jpg,  (4,4)
0300_c7_f0091101.jpg,  (8,4)
0341_c1_f0106021.jpg,  (1,4)
0390_c1_f0111618.jpg,  (4,4)
0468_c1_f0125465.jpg,  (3,4)
0482_c1_f0127523.jpg,  (6,3)
0543_c1_f0141101.jpg,  (3,3)
0581_c4_f0132958.jpg,  (6,4)


