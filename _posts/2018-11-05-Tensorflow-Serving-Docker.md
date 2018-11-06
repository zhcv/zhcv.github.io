---
layout: post
title: ""
categories: Tensorflow
tags: docker, deploy
author: ZhP
---

* content
{:toc}

# 使用 TensorFlow Serving 和 Docker 快速服务于机器学习


能够简单快捷地提供机器学习模型是从试验转向生产的关键挑战之一.服务机器学习模型就是采用经训练的模型并使其能够应对预测请求的过程.在生产中服务时.您需要确保您的环境可重现.强制隔离并且是安全的.为此.提供机器学习模型的最简单方法之一是就是将 TensorFlow Serving 与 Docker 结合起来. Docker 是一种将软件打包成单元(我们称之为容器)的工具.其中包含运行软件所需的一切.

## 使用 TensorFlow Serving 和 Docker 服务 ResNet

![TensorFlow Serving在Docker容器中运行]()
自 TensorFlow Serving 1.8 发布以来.我们一直在改进对 Docker 的支持. 我们现在提供 Docker images 用于 CPU 和 GPU 模型的服务和开发.为了解使用 TensorFlow Serving 部署模型究竟有多么容易.让我们尝试将 ResNet 模型投入生产. 此模型在 ImageNet 数据集上进行训练.并将 JPEG 镜像作为输入并返回镜像的分类类别.

我们的示例将假设您正在运行 Linux.不过它在 macOS 或 Windows 应该也可以运行.仅需少量修改.甚至不需要修改.

第一步安装 Docker CE. 这将为您提供运行和管理 Docker 容器所需的所有工具.

TensorFlow Serving 为其 ML 模型使用 SavedModel 格式.SavedModel 是一种语言中立的,可恢复的.密集的序列化格式.使更高级别的系统和工具能够生成.使用和转换 TensorFlow模型. 有几种方法可以导出 SavedModel(包括来自 Keras). 在本练习中.我们只需下载预先训练的 pre-trained ResNet SavedModel:

```bash
$ mkdir /tmp/resnet 
$ curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
```

我们现在应该在`/tmp/resnet`中有一个包含我们模型的文件夹.可以通过运行来验证这一点:

```bash
$ ls /tmp/resnet 
1538687457
```

现在我们有了模型.使用 Docker 服务就像拉来最新发布的 TensorFlow Serving 来服务环境镜像一样简单.并将其指向模型:

```
$ docker pull tensorflow/serving 
$ docker run -p 8501:8501  --name tfserving_resnet \ 
             --mount type=bind, source=/tmp/resnet.target=/models/resnet \ 
             -e MODEL_NAME=resnet -t tensorflow/serving &

... 
... main.cc:327]在0.0.0.0:8500运行ModelServer ...... 
... main.cc:337]导出HTTP / REST API:localhost:8501 ...
```

分解命令行参数.我们:

* -p 8501:8501 : 将容器的端口 8501(TensorFlow 服务响应 REST API 请求)发布到主机的端口 8501

* --name tfserving_resnet : 我们为容器创建名称为`tfserving_resnet`.这样稍后我们可以作参考

* --mount type=bind,source=/tmp/resnet,target=/models/resnet: 在容器(/models/resnet)上安装主机的本地目录(/tmp/resnet), 以便 TensorFlow 服务可以从容器内部读取模型.

* -e MODEL_NAME=resnet : 告诉 TensorFlow Serving 下载名为`resnet`的模型

* -t tensorflow/serving : 基于服务镜像`tensorflow/serving`运行 Docker 容器

接下来让我们下载 python 客户端脚本, 它将发送服务的模型镜像并获取预测, 我们还将测量服务器响应时间.

```shell
$ curl -o /tmp/resnet/resnet_client.py https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
```

此脚本将下载猫的镜像并在测量响应时间时将其重复发送到服务器.如脚本的主循环中所示:

```python
# The server URL specifies the endpoint of your server running the ResNet    
# model with the name "resnet" and using the predict interface.    
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'    

...    

# Send few actual requests and time average latency.    

total_time = 0    
num_requests = 10    
for _ in xrange(num_requests):    
    response = requests.post(SERVER_URL, data=predict_request)    

response.raise_for_status()    
total_time += response.elapsed.total_seconds()    
prediction = response.json()['predictions'][0]    

print('Prediction class: {}, avg latency: {} ms'.format(    

prediction['classes'], (total_time*1000)/num_requests))    
```

此脚本使用请求模块.因此如果您尚未安装.则需要安装它.通过运行此脚本.您应该看到如下所示的输出:

```bash
$ python /tmp/resnet/resnet_client.py
Prediction class: 282, avg latency: 185.644 ms
```

如您所见.使用 TensorFlow Serving 和 Docker 创建模型非常简单直白.您甚至可以创建自己的嵌入式模型的自定义 Docker 镜像. 以便更轻松地进行部署.

## 通过构建优化的 TensorFlow Serving 二进制文件来提高性能

既然我们在 Docker 中提供了一个模型.您可能已经注意到来自 TensorFlow Serving 的日志消息如下所示:

Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA

TensorFlow Serving 已发布的 Docker 镜像旨在竭尽所能来使用 CPU 架构.因此省略了一些优化以最大限度地提高兼容性.如果您没有看到此消息.则您的二进制文件可能已针对您的 CPU 进行了优化.
根据您的模型执行的操作.这些优化可能会对您的服务性能产生重大影响.值得庆幸的是.将您自己的优化服务镜像组合在一起非常简单.

首先.我们要构建 TensorFlow Serving 的优化版本.最简单的方法是构建官方的 TensorFlow Serving 开发环境 Docker 镜像.这具有为镜像构建的系统自动生成优化的 TensorFlow 服务二进制文件的良好特性.为了区分我们创建的镜像和官方镜像.我们将 $USER/ 添加到镜像名称之前.
让我们称这个开发镜像为 $USER/tensorflow-serving-devel:

```bash
$ docker build -t $USER/tensorflow-serving-devel \
    -f Dockerfile.devel \ 
    https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker
```

构建 TensorFlow 服务开发镜像可能需要一段时间.具体取决于计算机的速度. 完成之后让我们使用优化的二进制文件构建一个新的服务镜像. 并将其命名为 $USER/tensorflow-serving:

```bash
$ docker build -t $USER/tensorflow-serving \
               --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel \ 
               https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker
```

现在我们有了新的服务镜像.让我们再次启动服务器:

```bash
$ docker kill tfserving_resnet
$ docker run -p 8501:8501 --name tfserving_resnet \
             --mount type=bind,source=/tmp/resnet,target=/models/resnet \
             -e MODEL_NAME=resnet -t $USER/tensorflow-serving &
```

最后运行我们的客户端:
```python
$ python /tmp/resnet/resnet_client.py
Prediction class: 282, avg latency: 84.8849 ms
```

在我们的机器上.我们看到使用原生优化二进制文件.每次预测平均加速超过100毫秒(119%). 在不同的机器(和型号)上您可能会看到不同的结果.
最后.随意销毁 TensorFlow Serving 容器:

```command order stop better than kill```
```shell
$ docker stop tfserving_resnet
$ docker kill tfserving_resnet
```
``````````````````````````````````````````````````````````````````````````````
