# Surrogate-based Neural Network Structural Pruning

Supervisor: [Oleg Bakhteev, PhD](https://bahleg.site/publications)

|  |  Link |
| :---: | :---: |
| Paper |  [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/paper/main.pdf) |
| Code | [link](https://github.com/intsystems/Zero-shot-structural-pruning/tree/master/notebooks) |
| Slides | [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/slides/slides.pdf) |

## Abstract
This paper investigates the problem of structural pruning in neural networks. Structural pruning is the procedure of removing entire groups of parameters from a neural network, for example, filters in CNNs or skip connections. An appropriate pruning strategy can improve both generalization performance and inference efficiency. The main difficulty of structural pruning lies in the fact that when a layer is removed, all dependent layers must also be removed or consistently modified. 

The proposed method is based on the analysis of the deep learning computation graph and the estimation of the information flow propagated through it. The method enables the estimation of the importance of operations in the computation graph in a few-shot setting, i.e., using several forward passes of a data subset through the analyzed model. To demonstrate the effectiveness of the proposed method, we conduct multiple experiments on the MNIST dataset.

