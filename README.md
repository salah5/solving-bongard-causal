# solving-bongard-causal

<p align="center">
  <img src="https://www.foundalis.com/res/bps/bongard/p049.gif" />
</p>

## Step 1: Clone this repository

Python Version: 3.8

```
git clone https://github.com/salah5/solving-bongard-causal.git
```

then initialize stable-baselines3 submodule:

```
git submodule update --init
```

## Step 2: Install required modules

```
cd bongard_env
pip install -r requirements.txt
```

```
cd ..
cd sb3
pip install -e .
```

## Examples on how to train or evaluate an agent can be found in the corresponding jupyter notebooks

`bongard_env/train_agent.ipynb` shows how to train an agent on the BP environment.
`bongard_env/evaluate_model.ipynb` shows how to evaluate a trained agent.

You can visualize logs for training stats like avg reward, loss and episode length with tensorboard:<br />
`tensorboard --logdir=<log directory>`
