---
layout: post
title:  "Cblas-Lapack-Linux-Lib"
categories: Tools
tags: git,Github,tools
author: ZhP
---

* content
{:toc}


## Linux 安装 cblas, lapack, lapacke

1 确保机器上安装了gfortran编译器，如果没有安装的话，可以使用sudo apt-get install gfortran

2 下载blas, cblas, lapack 源代码， 这些源码都可以在 http://www.netlib.org 上找到，下载并解压。这里提供我安装时的下载链接.解压之后会有三个文件夹,BLAS, CBLAS, lapack-3.4.2

## [blas](http://www.netlib.org/blas/blas.tgz) [cblas](http://www.netlib.org/blas/blast-forum/cblas.tgz) [lapack](http://www.netlib.org/lapack/lapack-3.4.2.tgz)

3  这里就是具体的编译步骤

编译blas， 进入BLAS文件夹，执行以下几条命令
```shell
gfortran -c  -O3 *.f  # 编译所有的 .f 文件，生成 .o文件
ar rv libblas.a *.o  # 链接所有的 .o文件，生成 .a 文件
sudo cp libblas.a /usr/local/lib  # 将库文件复制到系统库目录
```

编译cblas， 进入CBLAS文件夹，首先根据你自己的计算机平台，将目录下某个 Makefile.XXX 复制为 Makefile.in , XXX表示计算机的平台，如果是Linux，那么就将Makefile.LINUX 复制为 Makefile.in，然后执行以下命令
```shell
cp ../BLAS/libblas.a  testing  # 将上一步编译成功的 libblas.a 复制到 CBLAS目录下的testing子目录
make # 编译所有的目录
sudo cp lib/cblas_LINUX.a /usr/local/lib/libcblas.a # 将库文件复制到系统库目录下
```

编译 lapack以及lapacke，这一步比较麻烦，首先当然是进入lapack-3.4.2文件夹，然后根据平台的特点，将INSTALL目录下对应的make.inc.XXX 复制一份到 lapack-3.4.2目录下，并命名为make.inc, 这里我复制的是 INSTALL/make.inc.gfortran，因为我这里用的是gfortran编译器。

修改lapack-3.4.2/Makefile, 因为lapack以来于blas库，所以需要做如下修改
```shell
# lib: lapacklib tmglib
lib: blaslib variants lapacklig tmglib
make # 编译所有的lapack文件
cd lapacke # 进入lapacke 文件夹，这个文件夹包含lapack的C语言接口文件
make # 编译lapacke
cp include/*.h /usr/local/include # 将lapacke的头文件复制到系统头文件目录
cd .. #返回到 lapack-3.4.2 目录
cp *.a /usr/local/lib   # 将生成的所有库文件复制到系统库目录
```

这里的头文件包括： ```lapacke.h, lapacke_config.h, lapacke_mangling.h, lapacke_mangling_with_flags.h lapacke_utils.h```

生成的库文件包括：`liblapack.a, liblapacke.a, librefblas.a, libtmglib.a`

至此cblas和lapack就成功安装到你的电脑上了。

测试：

可以到 LAPACKE 找测试代码，这里是lapacke的官方文档，比如以下代码：
```c
/* Calling DGELS using row-major order:  gfortran test.c -llapacke -llapack */
#include <stdio.h>  
#include <lapacke.h>  
   
int main (int argc, const char * argv[])  
{  
   double a[5*3] = {1,2,3,4,5,1,3,5,2,4,1,4,2,5,3};  
   double b[5*2] = {-10,12,14,16,18,-3,14,12,16,16};  
   lapack_int info,m,n,lda,ldb,nrhs;  
   int i,j;  
   
   m = 5;  
   n = 3;  
   nrhs = 2;  
   lda = 5;  
   ldb = 5;  
   
   info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'N', m, n, nrhs, a, lda, b, ldb);  
   
   for(i=0;i<n;i++)  
   {   
      for(j=0;j<nrhs;j++)  
      {   
         printf("%lf ",b[i+ldb*j]);  
      }   
      printf("\n");  
   }   
   return(info);  
}  
```

将上诉代码保存为test.c，编译时，别忘了使用gfortran，此外，还需要连接用到的库，编译上面的代码，应使用如下命令:

```shell
gfortran test.c -llapacke -llapack -lrefblas
```

如果能正常编译，即表示安装成功。如果要了解这段代码的具体含义，可以到 [LAPACKE](http://www.netlib.org/lapack/lapacke.html) 查看. [Fork-Address](https://blog.csdn.net/mlnotes/article/details/9676269#)
