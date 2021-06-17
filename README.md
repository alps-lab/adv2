### Adv<sup>2</sup>: Adversarial Attacks against both Prediction and Interpretation

#### Description:

This is the implementation and reproduction of the experiments of the paper "Interpretable deep learning under fire".

#### Requirements:

Pytorch

#### Folder Structure:

* exp_attacks_fixed: main attack files for CAM, RTS

* rev1: MASK attack implementation, attack for ISIC dataset

* rev2: GRAD attack implementation, attack for CIFAR10 dataset

* expr_detect: detecting adversarial examples

* expr_shape: random shape attack

* expr_transfer: measure transferability of our attack

Notice: the code name of our project during the development is ACID instead of Adv<sup>2</sup>. 


#### Citation:

If you use this codebase, please cite our paper:

```
@inproceedings{zhang:2020:adv2,
  title = {Interpretable deep learning under fire},
  author = {Zhang, Xinyang and Wang, Ningfei and Shen, Hua and Ji, Shouling and Luo, Xiapu and Wang, Ting},
  booktitle = {Proceedings of the USENIX Security Symposium (Security)},
  year = {2020},
}
```
