import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from model.encoder.encoder import create_encoder
import matplotlib.pyplot as plt
from model.feature_extractor import FeatureExtractor
import torch


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        input_size: int,
        output_size: int,
        encoder_type: str,
        learning_rate: float,
        lr_gamma: float
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters()
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
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate
        )
        scheduler = ExponentialLR(optimizer, gamma=self.lr_gamma)
        return [optimizer], [scheduler]

    def compute_loss(self, batch, mode):
        img, target, low_dim = batch
        pred = self(img, mode).squeeze()
        loss = F.mse_loss(pred, target)
        self.log_stat(f"{mode}_loss", loss)
        return loss, pred, target, low_dim

    def log_stat(self, name, stat):
        self.log(
            name,
            stat,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def plot_tsne(self, pred, target, low_dim, mode):
        f, axes = plt.subplots(1, 2, figsize=(10,5))
        axes[0].scatter(low_dim[:, 0], low_dim[:, 1], c=pred, cmap="RdBu_r")
        axes[1].scatter(low_dim[:, 0], low_dim[:, 1], c=target, cmap="RdBu_r")
        axes[0].set_title("Prediction")
        axes[1].set_title("Target")
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.tight_layout()
        self.trainer.logger.log_image(key=f"Recon {mode}", images=[f])
        plt.close()

    def training_step(self, batch, batch_idx):
        loss, pred, target, low_dim = self.compute_loss(batch, "train")
        self.plot_tsne(pred, target, low_dim, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred, target, low_dim = self.compute_loss(batch, "val")
        self.plot_tsne(pred, target, low_dim, "val")
        return pred, target, low_dim
