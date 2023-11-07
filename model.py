import open_clip
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics import R2Score


class BrainDiVEModule(pl.LightningModule):
    def __init__(
        self,
        subject: int,
        roi: str,
        hemisphere: str,
        num_voxels: int,
        learning_rate: float,
        weight_decay: float,
        init_diffuser: bool = False,
    ):
        super(BrainDiVEModule, self).__init__()
        self.save_hyperparameters()
        self.subject = subject
        self.roi = roi
        self.hemisphere = hemisphere
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_voxels = num_voxels

        # TODO: Add support for other embeddings + multiple embeddings
        self.clip, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k"
        )
        self.clip.requires_grad_(False)
        # TODO: Add support for other encoders
        self.encoder = nn.Linear(512, num_voxels)
        if init_diffuser:
            self.diffuser = None  # TODO: Implement image synthesis

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

    def forward(self, x):
        with torch.no_grad():
            x = self.clip.encode_image(x)
        x = self.encoder(x)
        return x

    # TODO: LR Scheduler
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def compute_loss(self, batch, mode):
        img, activation = batch
        pred = self(img)
        loss = F.mse_loss(pred, activation)
        if mode == "train":
            self.train_r2.update(pred, activation)
        elif mode == "val":
            self.val_r2.update(pred, activation)
        self.log_stat(f"{mode}_loss", loss)
        return loss, pred

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

    def on_training_epoch_end(self):
        self.log_stat("train_r2", self.train_r2.compute())
        self.train_r2.reset()

    def validation_step(self, batch, batch_idx):
        _, pred = self.compute_loss(batch, "val")
        return pred

    def on_validation_epoch_end(self):
        self.log_stat("val_r2", self.val_r2.compute())
        self.val_r2.reset()
