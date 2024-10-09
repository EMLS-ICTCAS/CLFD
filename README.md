# Continual Learning in the Frequency Domain

The official repository for NeurIPS'24 paper "Continual Learning in the Frequency Domain". We extended the original repo [DER++](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html) with our method.
## Setup

+ Use argument `--load_best_args` to use the best hyperparameters from the paper.
+ New models can be added to the `models/` folder.
+ New datasets can be added to the `datasets/` folder.

## How to run CLFD?
- Use `./utils/main.py` to run experiments.
- To reproduce the results in the paper run the following  

    `python utils/main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`
- For example, for dataset Seq-CIFAR10, run \
    `python utils/main.py --dataset seq-cifar10 --model er_clfd --buffer_size 50 --load_best_args`
  