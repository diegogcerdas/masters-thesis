import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from model.encoder.encoder import create_encoder
from model.feature_extractor import FeatureExtractor


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        input_size: int,
        output_size: int,
        encoder_type: str,
        learning_rate: float,
        lr_gamma: float,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters(ignore="feature_extractor")
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.lr_gamma = lr_gamma

        self.encoder = create_encoder(encoder_type, input_size, output_size)

    def forward(self, x, mode="val"):
        with torch.no_grad():
            x = self.feature_extractor(x, mode)
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def compute_loss(self, batch, mode):
        img, target, _ = batch
        pred = self(img, mode).squeeze()
        loss = F.mse_loss(pred, target)
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

    def validation_step(self, batch, batch_idx):
        _, pred = self.compute_loss(batch, "val")
        return pred
