---
layout: post
title:  "ZLMediaKit 流媒体服务搭建"
date:   2022-04-21 14:06:05
categories: stream media
tags: video 
excerpt: srs 流媒体服务学习笔记
---

* content
{:toc}

## [搭建ZLMediaKit流媒体服务框架](https://zhcv.github.io/2021/04/21/搭建ZLMediaKit流媒体服务/)

最近的工作需要架设一台流媒体服务器用于后续的业务开发。在git上找到了很好的开源框架[ZLMediaKit](https://link.segmentfault.com/?enc=zeN41PaCaVAEOUnmEfgCVA%3D%3D.ZJed3dF4CMfY%2FwZyT9uJfUpmFYQGrA%2FgIyA%2BTHjSc8vASXtxO6e8abQbP9%2Bs2t%2F8)，按照页面教程操作之后，总是不能成功编译webrtc模块，通过各种搜索和尝试，总算是搭建成功，现把过程分享如下，也给自己留个记录。
系统环境：`Ubuntu20.04.4`

### 1.获取代码
代码从git获取，如果没安装git，需要执行

```bash
sudo apt-get intall git
```

```shell
cd /opt
#拉取项目代码
git clone https://github.com/ZLMediaKit/ZLMediaKit.git
#国内用户推荐从同步镜像网站gitee下载 
git clone --depth 1 https://gitee.com/xia-chu/ZLMediaKit
cd ZLMediaKit
#不要忘了这句命令
git submodule update --init
```

### 2.安装编译器
```
#安装gcc
sudo apt-get install build-essential
#安装cmake
sudo apt-get install cmake
```

### 3.依赖库
1.openssl安装编译

```shell
#如果之前安装了可以先卸载:apt -y remove openssl
cd /opt
#从git下载
git clone https://github.com/openssl/openssl.git
#如果git下载太慢或者连接有问题（比如我），可以到gitee下载
git clone https://gitee.com/mirrors/openssl.git
#下面的依次执行
mv openssl openssl-src && cd openssl-src
./config --prefix=/opt/openssl
make -j4
sudo make install
cd /opt/openssl && cp -rf lib64 lib
```

2.libsrtp安装编译

```shell
cd /opt
git clone https://gitee.com/mirrors/cisco-libsrtp.git
cd cisco-libsrtp
./configure --enable-openssl --with-openssl-dir=/opt/openssl
make -j4
sudo make install
```

### 4.构建和编译ZLMediaKit
```shell
cd /opt/ZLMediaKit
mkdir build
cd build
cmake .. -DENABLE_WEBRTC=true -DOPENSSL_ROOT_DIR=/opt/openssl -DOPENSSL_LIBRARIES=/opt/openssl/lib 
cmake --build . --target MediaServer
```

### 5.补充操作
上一步操作执行后，运行服务成功但没有demo页面，发现对应的www文件夹以及ssl证书并未放入指定目录，需要进行补充操作

```shell
#把www文件夹复制到编译后的目录
cd /opt/ZLMediaKit
sudo cp -r www release/linux/Debug/
#把自带的ssl证书放到编译后的目录
sudo cp -r tests/default.pem release/linux/Debug/
```

### 6.启动服务
```shell
cd /opt/ZLMediaKit/release/linux/Debug
#通过-h可以了解启动参数
./MediaServer -h
#以守护进程模式启动
./MediaServer -d &
```

之后浏览器打开`https://你的服务器ip/webrtc`可以成功推流拉流
