DCGAN简单实现
===========
[『TensorFlow』DCGAN生成动漫人物头像_下](http://www.cnblogs.com/hellcat/p/8340491.html)  
根据网上开源项目以及自己的理解尝试出的DCGAN实现，重点在于熟悉TensorFlow对于这种特殊网络结构的控制流程学习

## 1、预处理
数据在预处理时采用了原像素数据除以127.5减去1的操作，使得输出值保持在0~1之间，这样配合sigmoid激活函数可以很好的模拟学习。

## 2、目录介绍
TFR_process.py，TFRecode数据生成以及处理脚本<br>
ops.py，层封装脚本<br>
DC_GAN.py，DC_GAN原型机<br>
utils.py，格式化绘图、保存图片函数，开源项目直接找来的<br>

## 3、实验步骤
先运行TFR_process.py产生TFRecord数据，
本部分涉及参量如下（位于TFR_process.py的起始位置）：
```Python
# 定义每个TFR文件中放入多少条数据
INSTANCES_PER_SHARD = 10000
# 图片文件存放路径
IMAGE_PATH = './Data_Set/cartoon_faces'
# 图片文件和标签清单保存文件
IMAGE_LABEL_LIST = 'images_&_labels.txt'
# TFR文件保存路径
TFR_PATH = './TFRecord_Output'
```

然后再运行DC_GAN.py使用前面的数据训练DC_GAN，
```Python
# TFR保存图像尺寸
IMAGE_HEIGHT = 48
IMAGE_WIDTH = IMAGE_HEIGHT
IMAGE_DEPTH = 3
# 训练batch尺寸
BATCH_SIZE = 64
```
当时为了方便，这些参量的设置也放在了TFR_process.py中，这是因为我的数据读取函数`batch_from_tfr`位于此文件中，该函数可以设置传入网络的图片大小。 
