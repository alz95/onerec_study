# 数据集下载建议（OneRec 2502.18965 复现）

你说你可以自己下载数据，这里给你一份**可直接用的链接清单**。  
> 说明：由于论文版本/开源实现可能会在预处理上有差异，最终请以论文正文/附录的实验设置为准。

## 1) MovieLens-1M
- 官网入口（GroupLens）：https://grouplens.org/datasets/movielens/1m/
- 直接下载 ZIP：https://files.grouplens.org/datasets/movielens/ml-1m.zip

## 2) Amazon Reviews（常见是 2018 版本的各子集）
- 论文页面（McAuley Lab）：https://nijianmo.github.io/amazon/index.html
- 2018 Reviews（JSON.GZ，总入口）：https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/
- 元数据入口：https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/metaFiles2/

> Amazon 子集很多（Books / Electronics / Beauty 等），请按论文中对应子集下载。

## 3) Yelp（常见是 Yelp2018 / Yelp Open Dataset）
- Yelp Open Dataset 官方页：https://business.yelp.com/data/resources/open-dataset/
- 直接下载入口（需要同意条款）：https://www.yelp.com/dataset

## 4) 若论文使用了公开预处理版本（推荐优先）
很多推荐系统论文会使用 RecBole/开源仓库里统一处理后的版本，优点是更容易对齐指标。
- RecBole 数据说明：https://recbole.io/dataset_list.html
- RecBole 仓库：https://github.com/RUCAIBox/RecBole

---

## 建议你按这个顺序下载
1. 先下 **MovieLens-1M**（小且快，方便先跑通）。
2. 再下论文主数据集（如 Amazon 对应子集）。
3. 再补 Yelp 等较大数据集做最终对齐。

## 你下载后给我这三样信息
1. 数据放置路径（例如 `/workspace/onerec_study/data/raw/ml-1m`）
2. 你实际下载的数据集名称（例如 `ml-1m`、`amazon-beauty`）
3. 文件列表（`ls` 输出）

我下一步就可以直接给你：
- 预处理脚本（按论文协议切分）
- 训练配置（路径 + 超参）
- 一键运行命令
