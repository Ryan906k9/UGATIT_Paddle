
# 文件说明：

1. utils.py   各种小工具，包括图像读写和预处理等

2. main.py   主程序

3. UGATIT.py   用于定义训练过程和测试过程，文件保存等

4. networks.py    用于定义网络结构

5. data 用于存放数据，解压后的数据分成了 trainA, trainB, testA, testB 这 4 个文件夹

6. results 文件夹，用于存放保存的模型文件，和训练过程中各种文件

7. results/data/img  训练过程中生成的效果图片

8. results/data/test  测试生成的效果图片，包含 A2B 和 B2A

9. results/data/model 保存的模型文件，这里保存的间隔比 results/ 目录下的长一些

10. 由于生成新版本的文件大小限制，最后的模型文件另外上传了数据集，需要挂载后解压到 results/ 文件夹

# 一、运行环境及依赖

* paddlepaddle-gpu==1.8.3.post97

* 图像处理库 imageio

* 数据集：https://aistudio.baidu.com/aistudio/datasetdetail/48778


# 二、数据处理

* 按照原论文的方式进行了各种数据处理，包括翻转，裁剪，归一化等

# 三、训练过程



* 首先以 0.0001 的 learning rate，训练 500000 iteration
* 然后 Poly nomial Decay 的方式降低学习率，训练 450000 个 iteration，学习率从 0.0001 降低到 0.00001
* 然后训练 45000 个 iteration，学习率从 0.00001 降低到 0.000001
* 最后以 0.000001 的学习率训练 5000 个 iteration




# 四、测试代码

* 测试代码做了修改，直接生成的是 A2B 和 B2A 的目标图片，而不是原程序中的那种拼接起来的对比图，这样可以用来算最后的 KID 和 FID 数据

* 测试结果存放的区域是 /results/data/test/





# 五、测试结果

## 1. 采用的测试代码

* 指标为 KID，原来论文里面用的是这个指标
* 测试代码地址：https://github.com/taki0112/GAN_metrics-Tensorflow

由于原代码是 tensorflow 1 的，没有和 paddle 转换的API 说明，所以没搞清楚怎么转到 paddle，于是在 Ai studio 上跑出结果后，在本地进行了测试

## 2. 测试结果

* 最终测试结果：

IS :  (1.4739025, 0.18158214)

FID :  0.8780633544921875

KID_mean :  0.9771001525223255

KID_stddev :  0.2940438920632005

mean_FID :  1.3247618560791015

mean_KID_mean :  11.141591203399003

mean_KID_stddev :  0.4219322588760406

* 原论文数据：

![](https://ai-studio-static-online.cdn.bcebos.com/9f385d1340b14a91a9fe9d671abc6ef93adf08a83b714aa195332de0d70369d0)

* 原论文项目地址：

https://github.com/znxlwm/UGATIT-pytorch
