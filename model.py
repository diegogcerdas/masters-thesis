import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BrainEncoderModule(pl.LightningModule):

    def __init__(self, learning_rate, weight_decay):
        super(BrainEncoderModule, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return not NotImplementedError()

    # TODO: LR Scheduler
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        return optimizer

    def compute_loss(self, batch, mode):
        return NotImplementedError()

    def log_stat(self, name, stat):
        self.log(
            name,
            stat,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred = self.compute_loss(batch, "val")
        return pred
