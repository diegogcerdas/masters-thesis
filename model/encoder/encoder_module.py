import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics import R2Score

from model.encoder.encoder import create_encoder
from model.feature_extractor import create_feature_extractor


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        subject: int,
        roi: str,
        hemisphere: str,
        num_voxels: int,
        feature_extractor_type: str,
        encoder_type: str,
        learning_rate: float,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters()
        self.subject = subject
        self.roi = roi
        self.hemisphere = hemisphere
        self.num_voxels = num_voxels
        self.feature_extractor_type = feature_extractor_type
        self.encoder_type = encoder_type
        self.learning_rate = learning_rate

        self.feature_extractor = create_feature_extractor(feature_extractor_type)
        self.encoder = create_encoder(
            encoder_type, self.feature_extractor.feature_size, num_voxels
        )

        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

    def forward(self, x, mode):
        with torch.no_grad():
            x = self.feature_extractor(x, mode)
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch, mode):
        img, activation = batch
        pred = self(img, mode)
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
