# DOGT: Double-Order Graph Transformers with Adaptive Node-Group Learning （submitted to ..）
Yupei Zhang, Xian Sheng, Xuequn Shang
## Overview
>Graph Transformers have recently attracted considerable attention for graph representation. Current methods often fail to consider hyper-order structures arising from implicit node groups within graphs. Moreover, many hyper-order structures of graphs are constructed by performing predefined heuristics or clustering algorithms, leading to disjointed optimization from downstream objectives. To this end, this paper introduces Double-Order Graph Transformers, i.e., DOGT, a new framework that adaptively integrates hyper-order features into graph representations through learnable node-group discovery. Specifically, DOGT employs a series of trainable mask matrices to dynamically infer hyper-order edges, i.e., node groups, and constructs a hyper-order graph. Hyper-order features are then extracted by performing a novel double-order attention and graph neural networks (GNNs) on the knitted hyper-graph. Finally, DOGT integrates the hyper-order features and the raw-order features derived from a parallel GNN branch into a feature pyramid, resulting in the double-order graph representation. On four real-world graph datasets, extensive experiments demonstrate that DOGT results in better performance on graph classification and regression than the state-of-the-art graph representation learning baselines.
## DOGT
<img width="874" height="300" alt="image" src="https://github.com/user-attachments/assets/e2f11d8b-8ebd-47c4-a88c-1800cebc0cb8" />

## Dependencies
>The code requires Python >= 3.9 and PyTorch >= 1.10.1.

## Run
>main_mnist.py for runing on mnist dataset

## Contact
>Please contact to ypzhaang@nwpu.edu.cn for any problems.
