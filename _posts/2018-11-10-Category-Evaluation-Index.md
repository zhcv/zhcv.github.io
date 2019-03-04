---
layout: post
title: "常用分类问题性能评价指标"
categories: AI, ML
tags: Tools
author: @Zhang
---

* content
{:toc}


```accuracy, recall, precision, confusion matrix F1-score, mAP```

在谈到分类评价问题是， 应首先要了解下几个基本概念，基本上所有的评价指标都是围绕着这几个
基本概念定义的:
* TP, True Positive   真阳性： 实际为正，预测为正
* FP, False Positive  假阳性： 实际为负，预测为正
* FN, False Negative  假阴性： 实际为正，预测为负
* TN，True Negative   真阴性： 实际为负，预测为负

P = TP + FN
N = TN + FP
正常情况: Accuracy = (TP + TF) / (P + N)
