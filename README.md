# LI-RARL

### Set up the environment

```
conda env create -f environment.yml
AutoROM
```

### Training

* Run the training interactively:

```
python main.py --config config.yaml --suite atari --env Atari-Pong --selector kmeans --forward dreamer --agent dqn --seed 43 --run test_run
```

* Submit batch training:

```
sbatch submit_training.sh
```