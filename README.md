# Gym-RL

## Train
Set the config file path in `train.py` and then run
```
python train.py
```
Then the results will be saved into `exp` directory.

## Log
```
tensorboard --logdir xxx(env_name)/xxx(timestamp)
```

## Infer
Set `conf_path` and `ckpt_path` in `play.py` and then run
```
python play.py
```