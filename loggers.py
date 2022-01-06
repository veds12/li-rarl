import wandb
from torch.utils.tensorboard import SummaryWriter

class WandbLogger:
    def __init__(self, config):
        self.project = config["project"]
        self.run = config["run"]
        wandb.init(
            settings=wandb.Settings(start_method="fork"), 
            run=self.run, 
            project=self.project, 
            config=config
            )

    def log(self, params, step):
        step = params[step]
        wandb.log(params, step=step)

class TensorboardLogger:
    def __init__(self, config):
        self.writer = SummaryWriter(config["logdir"])
        self.writer.add_text("config", f"{config}")

    def log(self, params, step):
        global_step = params[step]
        for key, value in params.items():
            self.writer.add_scalar(key, value, global_step=global_step)
