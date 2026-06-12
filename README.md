# Surrogate-based Neural Network Structural Pruning

Supervisor: [Oleg Bakhteev, PhD](https://bahleg.site/publications)

|  |  Link |
| :---: | :---: |
| Paper |  [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/paper/main.pdf) |
| Thesis |  [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/thesis/thesis.pdf) |
| Code | [link](https://github.com/intsystems/Zero-shot-structural-pruning/tree/master/src) |
| Slides | [link](https://github.com/intsystems/Zero-shot-structural-pruning/blob/master/slides/slides.pdf) |

## Abstract
This paper investigated the problem of structural pruning in neural networks. 
The proposed method is based on analyzing the deep learning computation graph and estimating the information flow propagated through it. 
The method enables the estimation of the importance of operations in a computation graph in a few-shot setting. 

Experiments have shown that the method is superior to surrogate models that do not use computation graph information. It is also possible to fine-tune the model after pruning, which can significantly improve its performance. Thus, the important role of computation graph information in structural pruning has been demonstrated. In future work, we plan to extend and generalize the method to channel-wise pruning (for example, filters in CNNs).

