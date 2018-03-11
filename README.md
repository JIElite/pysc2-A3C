# pysc2-A3C

I implemented a A3C agent by using pytorch to play Starcraft minigame, currently support CollectMineralShards. I this implementation, the shared model and local model are both in GPU.

## Dependency
- [Pytorch 0.3.1](http://pytorch.org/)
- [Pysc2](https://github.com/deepmind/pysc2)

## Issue
When I run 8 workers in parallel and put the models in GPU, the amount of GPU Memory usage for each worker is nearly 1200MB. How to reduce the memory usage?



## Reference
1. https://github.com/wing3s/pysc2-rl-mini
2. A3G: https://github.com/dgriff777/rl_a3c_pytorch
