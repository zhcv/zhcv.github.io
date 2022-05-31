---
layout: post
title: "LeNet 网络结构"
categories: deeplearning
tags: ai
author: ZhangHaipeng
---

* content
{:toc}


### 前言
[LeNet5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)通过巧妙的设计，利用卷积、参数共享、池化等操作提取特征，避免了大量的计算成本，最后再使用全连接神经网络进行分类识别，这个网络是卷积神经网络架构的起点，后续许多网络都以此为范本进行优化。Lenet 是一系列网络的合称，包括 Lenet1 - Lenet5，由 Yann LeCun 等人在 1990 年《Handwritten Digit Recognition with a Back-Propagation Network》中提出，是卷积神经网络的 HelloWorld。

### 一、Lenet5网络结构
Lenet是一个 7 层的神经网络，包含 3 个卷积层，2 个池化层，1 个全连接层。其中所有卷积层的所有卷积核都为 5x5，步长 strid=1，池化方法都为全局 pooling，激活函数为 Sigmoid，网络结构如下：
<img src="https://raw.githubusercontent.com/zhcv/zhcv.github.io/master/_posts/picture/LeNet-5.png">


特点：

1.相比MLP，LeNet使用了相对更少的参数，获得了更好的结果。

2.设计了maxpool来提取特征

### 二、Lenet的keras实现
如今各大深度学习框架中所使用的LeNet都是简化改进过的LeNet-5，和原始的LeNet有些许不同，把激活函数改为了现在很常用的ReLu。LeNet-5跟现有的conv->pool->ReLU的套路不同，它使用的方式是conv1->pool->conv2->pool2再接全连接层，但是不变的是，卷积层后紧接池化层的模式依旧不变。
```python
def LeNet():
    model = Sequential()
    # 原始的Lenet此处卷积核数量为6，且激活函数为线性激活函数
    model.add(Conv2D(32, (5,5), strides=(1,1), input_shape=(28,28,1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # 原始的Lenet此处卷积核数量为16，且激活函数为线性激活函数
    model.add(Conv2D(64, (5,5), strides=(1,1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
```

### 三、Lenet的pytorch实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes, grayscale=False): 
        """
        num_classes: 分类的数量
        grayscale：  是否为灰度图
        """
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale: # 可以适用单通道和三通道的图像
            in_channels = 1
        else:
            in_channels = 3

        # 卷积神经网络
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2)   # 原始的模型使用的是 平均池化
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),       # 这里把第三个卷积当作是全连接层了
            nn.Linear(120, 84), 
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)        # 输出 16*5*5 特征图
        x = torch.flatten(x, 1)     # 展平 （1， 16*5*5）
        logits = self.classifier(x) # 输出 10
        probas = F.softmax(logits, dim=1)
        return logits, probas


if __name__ == "__main__":
    num_classes = 10    # 分类数目
    grayscale = True    # 是否为灰度图
    data = torch.rand((1, 1, 32, 32))
    print("input data:\n", data, "\n")
    model = LeNet5(num_classes, grayscale)
    logits, probas = model(data)
    print("logits:\n",logits)
    print("probas:\n",probas)
```
  
