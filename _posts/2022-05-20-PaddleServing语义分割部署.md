---
layout: post
title: "PaddleServing图像语义分割部署实践"
categories: paddle
tags: segment
author: ZhangHaipeng
---

* content
{:toc}


## PaddleServing图像语义分割部署实践

#### 一、任务概述
本教程基于PaddleServing实现图像语义分割模型部署。首先我们会按照官方示例将部署流程跑一边，然后我们逐步调整代码和配置，基于更通用的PipeLine模式全流程实现抠图功能部署。

#### 二、官方示例部署
**2.1 安装PaddleServing**

从官网下载最新稳定离线版whl文件进行安装，各组件安装命令如下：
```shell
# 安装客户端：
pip install paddle-serving-client

# 安装服务器端（CPU或者GPU版二选一）：
安装CPU服务端：pip install paddle-serving-server
安装GPU服务端：pip install paddle-serving-server-gpu
安装工具组件：pip install paddle-serving-app
```
在安装时为了加速可以添加百度镜像源参数：`-i https://mirror.baidu.com/pypi/simple`

**2.2 导出静态图模型**

一般来说我们使用PaddlePaddle动态图训练出来的模型如果直接部署，其推理效率是比较低的。为了能够实现高效、稳定部署，我们需要将训练好的模型转换为静态图模型。

导出示例请参考官网说明。

这里我们使用官网示例给出的转换好的模型进行操作。

下载静态图模型：

```shell
wget https://paddleseg.bj.bcebos.com/dygraph/demo/bisenet_demo_model.tar.gz
tar zxvf bisenet_demo_model.tar.gz
```

解压后看到模型文件夹中内容如下所示：
然后我们准备一张用于测试的街景图像：

![img](https://img-blog.csdnimg.cn/f2f4b2e8b2ac483b953222bd0d79daa3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6ZKx5b2sIChRaWFuIEJpbik=,size_20,color_FFFFFF,t_70,g_se,x_16)
到这里，需要的部署数据都准备好了。

**2.3 转换为serving模型**

为了能够使用Paddle Serving工具实现AI服务器云部署，我们需要前面准备好的静态图模型转换为Paddle Serving可以使用的部署模型。
我们将使用`paddle_serving_client.convert`工具进行转换，具体命令如下：
```shell
python -m paddle_serving_client.convert \
	--dirname ./bisenetv2_demo_model \
	--model_filename model.pdmodel \
	--params_filename model.pdiparams
```
执行完成后，当前目录下的serving_server文件夹保存服务端模型和配置，serving_client文件夹保存客户端模型和配置，如下图所示：
![img](https://img-blog.csdnimg.cn/a74d3ffad39c44e3a79a12f9f5ff135a.png)

**2.4 启动服务**

按照官方示例，我们使用paddle_serving_server.serve的RPC服务模式，详细信息请参考文档。（需要注意的是，这种模式本质上是C/S架构，优势是响应快，缺点是在客户端需要安装相应的库并需要编写预处理代码）

我们在服务器端使用`27008`端口。
```shell
python3 -m paddle_serving_server.serve \
	--model serving_server \
	--thread 10 \
	--port 27008 \
	--ir_optim
```
启动后如果我们机器上没有安装对应版本的tensorrt，那么启动会出现如下错误：
error while loading shared libraries: libnvinfer.so.6: cannot open shared object file: No such file or directory
我们需要下载tensorrt库并将其添加到自己的环境变量去（注意tensortrt版本要与我们安装的paddle-serving-server-gpu版本一致）。相关解决方案请参考博客。安装完成后需要导入环境变量：

打开并编辑bashrc文件：`vim ~/.bashrc`
在文件最后添加：
```shell
export LD_LIBRARY_PATH=/home/suser/copy/TensorRT-6.0.1.8/lib:$LD_LIBRARY_PATH
```

保存修改后把相关文件进行拷贝：
```shell
sudo cp TensorRT-6.0.1.8/targets/x86_64-linux-gnu/lib/libnvinfer.so.6  /usr/lib/
```

最后使用下面的命令使其生效：
`source ~/.bashrc`
重新启动服务端就可以正常跑起来了。

**2.5 客户端请求**

客户端采用python脚本进行访问请求。
完整请求代码如下：
```python
import os
import numpy as np
import argparse
from PIL import Image as PILImage
from paddle_serving_client import Client
from paddle_serving_app.reader import Sequential, File2Image, Resize, CenterCrop
from paddle_serving_app.reader import RGB2BGR, Transpose, Div, Normalize


def get_color_map_list(num_classes, custom_color=None):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map
    
def get_pseudo_color_map(pred, color_map=None):
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask
    
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        "--serving_client_path",
        help="The path of serving_client file.",
        type=str,
        required=True)
    parser.add_argument(
        "--serving_ip_port",
        help="The serving ip.",
        type=str,
        default="127.0.0.1:9292",
        required=True)
    parser.add_argument(
        "--image_path", help="The image path.", type=str, required=True)
    return parser.parse_args()
    
def run(args):
    client = Client()
    client.load_client_config(
        os.path.join(args.serving_client_path, "serving_client_conf.prototxt"))
    client.connect([args.serving_ip_port])

    seq = Sequential([
        File2Image(), RGB2BGR(), Div(255),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], False), Transpose((2, 0, 1))
    ])

    img = seq(args.image_path)
    fetch_map = client.predict(
        feed={"x": img}, fetch=["save_infer_model/scale_0.tmp_1"])

    result = fetch_map["save_infer_model/scale_0.tmp_1"]
    color_img = get_pseudo_color_map(result[0])
    color_img.save("./result.png")
    print("The segmentation image is saved in ./result.png")
    
if name == 'main':
    args = parse_args()
    run(args)
```

然后使用下面的命令启动：
```shell
python test.py \
    --serving_client_path serving_client \
    --serving_ip_port 127.0.0.1:27008 \
    --image_path cityscapes_demo.png
```

运行后可能会出现下面的错误：
````shell
libcrypto.so.10: cannot open shared object file: No such file or directory
````

解决方案如下：
```bash
wget https://paddle-serving.bj.bcebos.com/others/centos_ssl.tar 
tar xf centos_ssl.tar 
rm -rf centos_ssl.tar
sudo mv libcrypto.so.1.0.2k /usr/lib/libcrypto.so.1.0.2k
sudo mv libssl.so.1.0.2k /usr/lib/libssl.so.1.0.2k 
sudo ln -sf /usr/lib/libcrypto.so.1.0.2k /usr/lib/libcrypto.so.10 
sudo ln -sf /usr/lib/libssl.so.1.0.2k /usr/lib/libssl.so.10 
sudo ln -sf /usr/lib/libcrypto.so.10 /usr/lib/libcrypto.so
sudo ln -sf /usr/lib/libssl.so.10 /usr/lib/libssl.so
```

修改后重新执行客户端请求代码，结果如下图所示：
```shell
I0423 10:45:54.155158 20937 naming_service_thread.cpp:202] brpc::policy::ListNamingService("127.0.0.1:27008"): added 1

I0423 10:46:01.689276 20937 general_model.cpp:490] [client]logid=0,client_cost=7292.98ms,server_cost=6954.06ms.
The segmentation image is saved in ./result.png
```

执行完成后，分割的图片保存在当前目录的`result.png`。
分割结果如下图所示：

![img](https://img-blog.csdnimg.cn/2fd46d5e6ff342c69d56c8722ca2fd25.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6ZKx5b2sIChRaWFuIEJpbik=,size_20,color_FFFFFF,t_70,g_se,x_16)

想要彻底停止服务可以使用下面的命令：

```shell
ps -ef | grep serving | awk '{print $2}' | xargs kill -9
ps -ef | grep web_service | awk '{print $2}' | xargs kill -9
```
从整个执行上来分析，这种基于RPC的方式有个明显的缺点，就是需要客户端来实现所有的预处理和后处理操作。这对于跨语言的应用任务来说是比较麻烦的，例如我们如果采用java作为前台语言，那么就只能使用java来执行图像相关预处理和后处理。为了解决这个问题，`paddle serving`提供了`java`版的客户端，其本质是一个封装好的基于java的图像预处理工具。前端程序员还是需要手工编写客户端代码，协调合作时比较麻烦。

下面我们将使用另一种`pipeline`的方法，所有的预处理和后处理也一起交给服务端去做，这样就彻底跟前端功能剥离开来，前后端之间通过http接口进行通讯。这种方式相对于RPC模式来说速度会慢一些，但是很显然，其通用性更好。

下面我们就使用一个更加具体的抠图任务来实现整个的PipeLine部署。

#### 三、基于PipeLine的抠图功能部署
**3.1 基于深度学习的抠图功能测试**

**3.1.1 算法库下载**

首先从github官网下载最新的paddleseg套件，也可以从我的gitee镜像上下载（速度会快一些）：

```shell
git clone https://gitee.com/binghai228/PaddleSeg.git
cd PaddleSeg
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

python setup.py install
```

注意如果本地安装失败，也可以使用在线安装方式：
```shell
cd PaddleSeg
pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple
pip install paddleseg==2.5.0

cd到Matting文件夹中：
cd PaddleSeg/Matting
```

这个文件夹下面就是`PaddleSeg`官方在维护的抠图算法套件。
**3.1.2 抠图算法说明**
Matting（精细化分割/影像去背/抠图）是指借由计算前景的颜色和透明度，将前景从影像中撷取出来的技术，可用于替换背景、影像合成、视觉特效，在电影工业中被广泛地使用。影像中的每个像素会有代表其前景透明度的值，称作阿法值（Alpha），一张影像中所有阿法值的集合称作阿法遮罩（Alpha Matte），将影像被遮罩所涵盖的部分取出即可完成前景的分离。

相关功能实现效果如下：

![img](https://img-blog.csdnimg.cn/58bb571ab02f4ce2957d90efa77e6411.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6ZKx5b2sIChRaWFuIEJpbik=,size_20,color_FFFFFF,t_70,g_se,x_16)

PaddleSeg套件提供多种场景人像抠图模型, 可根据实际情况选择相应模型。这里我们选择PP-Matting-512模型进行部署应用。读者也可以参照官网教程自行训练模型，然后转为静态图模型使用。本教程更偏重算法部署，对于算法原理和训练本教程不再深入阐述，对深度学习抠图有兴趣的读者可以参考我的另一篇博客了解相关算法原理。

**3.1.3 抠图算法测试**

首先下载训练好的模型。如下图所示：
![img](https://img-blog.csdnimg.cn/d7039f9e2b7942f7b6262f76181aa2ed.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA6ZKx5b2sIChRaWFuIEJpbik=,size_19,color_FFFFFF,t_70,g_se,x_16)

模型下载后解压放置在Matting/data文件夹下。
然后我们下载[PPM-100](https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip)数据集用于后续测试。下载下来后解压放置在Matting/data目录下。

最终我们可以使用下面的脚本命令进行测试：
```python
python deploy/python/infer.py \
    --config data/pp-matting-hrnet_w18-human_512/deploy.yaml \
    --image_path data/PPM-100/val/fg/ \
    --save_dir output/results
```
推理完成后在output/results目录下保存了抠图后的测试图像结果。
部分效果如下：

 <img src="https://img-blog.csdnimg.cn/2f4126a9caf3424483d76397fba6efe8.png" width="200"><img src="https://img-blog.csdnimg.cn/8dd6e25e912b4d079008105b94260957.png" width="200">

从效果上分析，整体抠图性能还是比较好的。
当然，对于一些复杂的照片抠图效果还是有待再提高的，例如下面的示例：

<img src="https://img-blog.csdnimg.cn/5ccade8820a54449903f566173074186.png" width="300" /><img src="https://img-blog.csdnimg.cn/14223580645c459bb3648f445c6f0429.png" width="300" />


##### 3.2 基于PipeLine的Serving部署
**3.2.1 转换为serving部署模型**

使用`paddle_serving_client.convert`工具进行转换，具体命令如下：

```python
python -m paddle_serving_client.convert \
    --dirname ./data/pp-matting-hrnet_w18-human_512 \
    --model_filename model.pdmodel \
    --params_filename model.pdiparams
```
执行完成后，当前目录下的serving_server文件夹保存服务端模型和配置，serving_client文件夹保存客户端模型和配置，如下图所示：
我们打开serving_server_conf.prototxt文件，其内容如下所示：
```protobuf
feed_var {
    name: "img"
    alias_name: "img"
    is_lod_tensor: false
    feed_type: 1
    shape: 3
}

fetch_var {
    name: "tmp_75"
    alias_name: "tmp_75"
    is_lod_tensor: false
    fetch_type: 1
    shape: 1
}
```
根据这个文件，我们在写部署代码的时候需要注意对应的输入、输出变量名称，这里输入变量名为img,输出变量名为tmp_75。
**3.2.2 设置config.yml部署配置文件**

在当前目录下新建config.yml文件，内容如下：
```yaml
dag:
  #op资源类型, True, 为线程模型；False，为进程模型
  is_thread_op: false
  #使用性能分析, True，生成Timeline性能数据，对性能有一定影响；False为不使用
  use_profile: false
  # tracer:
  #   interval_s: 30
#http端口, rpc_port和http_port不允许同时为空。当rpc_port可用且http_port为空时，不自动生成http_port
http_port: 27008

#worker_num: 2      #最大并发数。当build_dag_each_worker=True时, 框架会创建worker_num个进程，每个进程内构建grpcSever和DAG

#build_dag_each_worker, False，框架在进程内创建一条DAG；True，框架会每个进程内创建多个独立的DAG
build_dag_each_worker: false

#rpc端口, rpc_port和http_port不允许同时为空。当rpc_port为空且http_port不为空时，会自动将rpc_port设置为http_port+1
#rpc_port: 27009

op:
  matting:
    #并发数，is_thread_op=True时，为线程并发；否则为进程并发
    concurrency: 4
    local_service_conf:
      #client类型，包括brpc, grpc和local_predictor.local_predictor不启动Serving服务，进程内预测
      client_type: local_predictor
      # device_type, 0=cpu, 1=gpu, 2=tensorRT, 3=arm cpu, 4=kunlun xpu
      device_type: 1
      #计算硬件ID，当devices为""或不写时为CPU预测；当devices为"0", "0,1,2"时为GPU预测，表示使用的GPU卡
      devices: '0,1,2,3'
      #Fetch结果列表，model中fetch_var的alias_name为准, 如果没有设置则全部返回
      fetch_list:
      - tmp_75
      #模型路径
      model_config: serving_server
```
如果要部署自己的模型请根据注释结合自己需要部署的模型参数对照着进行修改。

**3.2.3 编写服务端脚本文件**

新建`web_service.py`文件，内容如下：
```python
import numpy as np
import cv2
from paddle_serving_app.reader import *
import base64
from paddle_serving_server.web_service import WebService, Op

class MattingOp(Op):
    '''
    定义抠图算子
    '''
    def init_op(self):
        '''
        初始化
        '''
        self.img_preprocess = Sequential([
            #BGR2RGB(), 
            Div(255.0),
            #Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False),
            Resize(512), Transpose((2, 0, 1))
        ])
        self.ref_size = 512
        self.img_width = self.ref_size
        self.img_height = self.ref_size   
 
    def preprocess(self, input_dicts, data_id, log_id):
        '''
        预处理
        '''
        (_, input_dict), = input_dicts.items()
        imgs = []
        for key in input_dict.keys():
            # 解码图像
            data = base64.b64decode(input_dict[key].encode('utf8'))
            data = np.fromstring(data, np.uint8)
            im = cv2.imdecode(data, cv2.IMREAD_COLOR)
            self.im = im
            self.img_height,self.img_width,_ = im.shape
            # 短边对齐512，长边设置为32整数倍(根据算法模型要求)
            im_h, im_w, _ = im.shape  
            if im_w >= im_h:
                im_rh = self.ref_size
                im_rw = int(im_w *1.0 / im_h * self.ref_size)
            elif im_w < im_h:
                im_rw = self.ref_size
                im_rh = int(im_h *1.0 / im_w * self.ref_size)      
            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = cv2.resize(im,(im_rw,im_rh))

            # cv2转tensor
            im = self.img_preprocess(im)
            imgs.append({
                "img": im[np.newaxis, :],
                # "im_shape":np.array(list(im.shape[1:])).reshape(-1)[np.newaxis, :],
                # "scale_factor": np.array([1.0, 1.0]).reshape(-1)[np.newaxis, :],
            })

        # 准备输入数据
        feed_dict = {
            "img": np.concatenate(
                [x["img"] for x in imgs], axis=0),
            # "im_shape": np.concatenate(
            #     [x["im_shape"] for x in imgs], axis=0),
            # "scale_factor": np.concatenate(
            #     [x["scale_factor"] for x in imgs], axis=0)
        }
        #for key in feed_dict.keys():
        #    print(key, feed_dict[key].shape)
        return feed_dict, False, None, ""
 
    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        '''
        后处理
        '''
        # 取出掩码图
        alpha = fetch_dict["tmp_75"]
        alpha = alpha.squeeze(0).squeeze(0)
        alpha = (alpha * 255).astype('uint8')  
        alpha = cv2.resize(
                    alpha, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        alpha = alpha[:, :, np.newaxis]
        clip = np.concatenate([self.im, alpha], axis=-1)  
        print(clip.shape)
        _, buffer_img = cv2.imencode('.png', clip)  # 在内存中编码为png格式
        img64 = base64.b64encode(buffer_img)  
        img64 = str(img64, encoding='utf-8')  # bytes转换为str类型 

        #封装成字典返回
        res_dict = {
            "alpha":img64
        }
        return res_dict, None, ""
 
class MattingService(WebService):
    '''
    定义服务
    '''
    def get_pipeline_response(self, read_op):
        matting_op = MattingOp(name="matting", input_ops=[read_op])
        return matting_op
      
# 创建服务
matting_service = MattingService(name="matting")
# 加载配置文件
matting_service.prepare_pipeline_config("config.yml")
# 启动服务
matting_service.run_service()
```
上述代码做了注释，读者可以自行阅读分析。

最后使用下面的命令启动服务：

```shell
python web_service.py
```

**3.2.4 客户端调用**

这里需要注意，由于我们采用了Pipeline模式，所有的图像预处理和后处理操作都放在了服务端，因此，客户端不需要加载额外的库，也不需要进行相关图像预处理代码编写。因此，我们可以采用任何客户端方式（浏览器、脚本、移动端等），只需要按照http restful协议传送相关json数据即可。

本文为了简单，采用python脚本来作为客户端（也可以仿照这个脚本使用postman进行测试）。新建脚本文件`pipeline_http_client.py`，具体代码如下：

```python
import numpy as np
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 定义http接口
url = "http://127.0.0.1:27008/matting/prediction"
# 打开待预测的图像文件
with open('./data/PPM-100/val/fg/1.jpg', 'rb') as file:
    image_data1 = file.read()

# 采用base64编码图像文件
image = cv2_to_base64(image_data1)
# 按照特定格式封装成字典
data = {"key": ["image"], "value": [image]}

# 发送请求
r = requests.post(url=url, data=json.dumps(data))

# 解析返回值
r = r.json()

# 解码返回的图像数据
img = r"value"
img = bytes(img, encoding='utf-8')  # str转bytes
img = base64.b64decode(img)  # base64解码
img = np.asarray(bytearray(img), dtype="uint8")
img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)

# 保存图像到本地
if img is None:
    print('call error')
else:
	cv2.imwrite('result.png',img)
	print('完成')
```
执行预测：`python3 pipeline_http_client.py`

#### 四、小结
本教程以PaddleServing部署为目标，以语义分割（抠图）案例贯穿整个部署环节，最终成功实现服务器线上部署和调用。通过本教程的学习，可以快速将训练好的深度学习模型进行上线，同时具备良好的稳定性。
