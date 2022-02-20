import wandb
from torch.utils.tensorboard import SummaryWriter

class WandbLogger:
    def __init__(self, config):
        self.project = config["project"]
        self.run = config["run"]
        wandb.init(
            settings=wandb.Settings(start_method="fork"), 
            name=self.run, 
            project=self.project, 
            config=config
            )

    def log(self, step, **kwargs):
        step = kwargs[step]
        wandb.log(kwargs, step=step)

class TensorboardLogger:
    def __init__(self, config):
        self.writer = SummaryWriter(config["logdir"])
        self.writer.add_text("config", f"{config}")

    def log(self, step, **kwargs):
        print(kwargs)
        global_step = kwargs[step]
        for key, value in kwargs.items():
            print(f"key is is {key}, value is {value}, global_step is {global_step}")
            self.writer.add_scalar(key, value, global_step=global_step)
