---
layout: post
title: "IQA"
categories: project
tags: project, working
author: ZhP
---

* content
{:toc}

## 图像质量主观评价


## 图像质量客观评价
图像质量客观评价可分为全参考（Full-Reference,FR），部分参考（Reduced-Reference,RR）和 
无参考（No-Reference,NR）三种类型。



1）熵
熵是指图像的平均信息量，它从信息论的角度衡量图像中信息的多少，图像中的信息熵越大，说明图像包含的信息越多。假设图像中各个像素点的灰度值之间是相互独立的，图像的灰度分布为p={p1,p2,…,pi,…,pn},其中pi表示灰度值为i的像素个数与图像总像素个数之比，而n为灰度级总数，其计算公式为：
E = -
其中，P(l)为灰度值l在图像中出现的概率，L为图像的灰度级，对于256灰度等级的图像而言，L=255.


在生产环境上迁移GitLab的目录需要注意一下几点：

1. 目录的权限必须为755或者775

2. 目录的用户和用户组必须为git:git

3. 如果在深一级的目录下，那么git用户必须添加到上一级目录的账户。

4. 很多文章说修改/etc/gitlab/gitlab.rb这个文件里面的`git_data_dirsb`变量，其实没必要，只需要使用软链接改变原始目录/var/opt/gitlab/git-data更好一些.

5. 注意：迁移前的版本和迁移后的版本必须保持一致, 如果迁移后的版本是高版本, 那么现在原版本做升级后再迁移.

迁移方法:
```shell
# 停止服务
gitlab-ctl stop

# 备份目录
mv /var/opt/gitlab/git-data{,_bak}

# 新建新目录
mkdir -p /data/service/gitlab/git-data

# 设置目录权限
chown -R git:git /data/service/gitlab
chmod -R 775 /data/service/gitlab

# 同步文件，使用rsync保持权限不变
rsync -av /var/opt/gitlab/git-data_bak/repositories /data/service/gitlab/git-data/

# 创建软链接
ln -s /data/service/gitlab/git-data /var/opt/gitlab/git-data

# 更新权限
gitlab-ctl upgrade

# 重新配置
gitlab-ctl reconfigure

# 启动gitlab服务
gitlab-ctl start
```
