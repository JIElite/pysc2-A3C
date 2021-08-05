# Neural Grafting Network


## DEMO
https://www.youtube.com/watch?v=u-AF5hc8l-o


## Environment Setup

Our experiments are based on StarCraftII 3.16, PySC2, Pytorch, and handcrafted mini-games. We highly recommend you install the package through conda.

#### STEP1: Install StarCraftII 3.16

Please refer to the [GitHub page](https://github.com/deepmind/pysc2#get-starcraft-ii) of PySC2 to download and install StarCraftII 3.16.

NOTICE: If you are using AMD CPUs, it may not be compatible with our repo. When I used AMD 5950X to install the StarCraftII 3.16 and our handcrafted mini-games, it didn't work.


#### STEP2: Create a conda environment and install dependent packages

- Create a conda environment for Python3.6
```
conda create --name pysc2_torch18_py36 python=3.6
conda activate pysc2_torch18_py36
```

- Install Pytorch 1.8.1 LTS (Here, we take Ubuntu as an example)
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
```

- Install PySC2 1.2
```
pip install pysc2==1.2
pip install pygame==1.9.6 ## Please refer to the troubleshooting as follows for more information
```

TROUBLE SHOOTING: [pygame.error: Unable to make GL context current](https://github.com/deepmind/pysc2/issues/308)

#### STEP3: Install the handcrafted mini-games

Please copy the *.SC2Map files from `Maps/mini_games/` to `StarCraftII/Maps/mini_games/`, and then go to the `pysc2

```
# TODO
```

#### STEP4: Verify the installation

Just play with it!
```
python -m pysc2.bin.play --map 
```

## Handcrafted Mini-Games

```
# TODO
```

## Train Reinforcement Learning Agent

```
# TODO
```

## Results

```
# TODO
```

## Reference
1. https://github.com/wing3s/pysc2-rl-mini
2. A3G: https://github.com/dgriff777/rl_a3c_pytorch