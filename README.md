# 📍 Dataset-Pruing-FL-FedCS
Official PyTorch implementation of paper (CVPR 2025) 🤩
"FedCS：Coreset Selection for Federated Learning" 
>[Chenhe Hao](https://github.com/xrosssaber12306), [Weiying Xie](https://scholar.google.com/citations?user=y0ha5lMAAAAJ&hl=zh-CN), [Daixun Li](https://scholar.google.cz/citations?user=gaiP4-IAAAAJ&hl=zh-CN&oi=ao), [Haonan Qin](https://scholar.google.cz/citations?hl=zh-CN&user=ePGTHqkAAAAJ), [Hangyu Ye](https://github.com/Yehangyu-XD), [Leyuan Fang](https://scholar.google.cz/citations?user=Gfa4nasAAAAJ&hl=zh-CN&oi=ao), [Yunsong Li](https://ieeexplore.ieee.org/author/37292407800)<br>
>XDU and HNU
![image](https://github.com/xrosssaber12306/Dataset-Pruing-FL-FedCS/blob/main/imgs/Framework.png)
Here is the [video](https://youtu.be/j9tpus2MHTg) 📺.
## 📝 Abstract
Federated Learning (FL) is an emerging direction in distributed machine learning that enables jointly training a model without sharing the data. However, as the size of datasets grows exponentially, computational costs of FL increase. In this paper, we propose the first Coreset Selection criterion for Federated Learning (FedCS) by exploring the Distance Contrast (DC) in feature space. Our FedCS is inspired by the discovery that DC can indicate the intrinsic properties inherent to samples regardless of the networks. Based on the observation, we develop a method that is mathematically formulated to prune samples with high DC. The principle behind our pruning is that high DC samples either contain less information or represent rare extreme cases, thus removal of them can enhance the aggregation performance. Besides, we experimentally show that samples with low DC usually contain substantial information and reflect the common features of samples within their classes, such that they are suitable for constructing coreset. With only two time of linear-logarithmic complexity operation, FedCS leads to significant improvements over the methods using whole dataset in terms of computational costs, with similar accuracies. For example, on the CIFAR-10 dataset with Dirichlet coefficient $\alpha=0.1$, FedCS achieves 58.88\% accuracy using only 44\% of the entire dataset, whereas other methods require twice the data volume as FedCS for same performance.

## 👩🏻‍💻 Usage and Examples
Use following steps you can reproduce FedCS on CIFAR10, CIFAR100, CINIC-10, TINY-IMAGENET. Here we use CIFAR10 as an example, the detailed training setting can be found in our paper.
### Environment 🌏
We conduct our experiments on 3090 GPUs in an environment configured as follows:
torch >= 1.8
torchvision >= 0.9
numpy >= 1.19

Readers do not need to replicate our setup exactly.
## Configurations 🧮
* See `config.yaml`
  
## Run 🏃🏻‍♀️
* `python main.py`

## 🙋🏻‍♀️ Citation

If you find our codes useful for your research, please cite our paper. 🤭

```
@inproceedings{Hao2025FedCS,
  title={FedCS：Coreset Selection for Federated Learning},
  author={Chenhe, Hao and Weiying, Xie and Daixun, Li and Haonan, Qin and Hangyu, Ye and Leyuan, Fang and Yunsong, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025},
}
```
Mario Kart world is a good game.
![image](https://github.com/xrosssaber12306/Dataset-Pruing-FL-FedCS/blob/main/imgs/Mariokart.jpg)
ps: Nintendo, give us back the looped tracks! We're tired of all these straight-line maps!
