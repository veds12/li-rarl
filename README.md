# Latent Imagination Based Retrieval Augmented Reinforcement Learning

* Setup the environment
```
conda env create -f environment.yml
AutoRAM
```

* Run dreamerv2 for pong
```
python dreamer-torch/dreamer.py --logdir ./logs/atari_pong/dreamerv2/1 --configs defaults atari --task atari_pong
```

* Submit batch training for pong
```
sbatch submit_training.sh
```