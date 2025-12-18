# Surrogate-based Neural Network Structural Pruning

|  |  Link |
| :---: | :---: |
| Paper |  [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/paper/main.pdf) |
| Code | [link](https://github.com/intsystems/Zero-shot-structural-pruning/tree/master/notebooks) |
| Slides | [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/slides/slides.pdf) |

## Abstract
The paper investigates the problem of structural pruning of models. Structural pruning is the process of removing groups of unimportant weights from a neural network, for example, filters in CNN or skip-connections. Proper pruning strategy leads to improvement of both generalizing ability and inference performance. Main difficulty of structural pruning is that when one layer of the network is removed, its dependent layers should also be removed. The proposed method is based on the deep learning computation graph analyzing and estimation of information flow transferred through it. The method enables estimation of the importance of operations in a computation graph in a zero-shot mode, i.e., using only a single pass of a subset of data through the analyzed model.p The basic idea [TODO]. To demonstrate the performance of the proposed method we conduct multiple experiments on synthetic data, CIFAR-10 and Wikitext dataset [TODO].

