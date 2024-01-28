import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics.functional import r2_score

from model.encoder.encoder import create_encoder
from model.feature_extractor import create_feature_extractor
import matplotlib.pyplot as plt


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
        lr_gamma: float
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
        self.lr_gamma = lr_gamma

        self.feature_extractor = create_feature_extractor(feature_extractor_type)
        self.encoder = create_encoder(
            encoder_type, self.feature_extractor.feature_size, num_voxels
        )

    def forward(self, x, mode):
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
        img, activation, _ = batch
        pred = self(img, mode).squeeze()
        loss = F.mse_loss(pred, activation)
        metric = r2_score(pred, activation)
        self.log_stat(f"{mode}_loss", loss)
        self.log_stat(f"{mode}_r2_score", metric)
        return loss, pred, activation

    def log_stat(self, name, stat):
        self.log(
            name,
            stat,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def plot_recon(self, pred, acts, mode):
        preds = pred.squeeze().detach().cpu().numpy()
        acts = acts.squeeze().detach().cpu().numpy()
        f, axes = plt.subplots(1, 2, figsize=(10,4))
        axes[0].plot(acts)
        axes[1].plot(preds)
        axes[0].set_ylim(-2,2)
        axes[1].set_ylim(-2,2)
        plt.tight_layout()
        self.trainer.logger.log_image(key=f"Recon {mode}", images=[f])
        plt.close()

    def training_step(self, batch, batch_idx):
        loss, pred, acts = self.compute_loss(batch, "train")
        self.plot_recon(pred, acts, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        _, pred, acts = self.compute_loss(batch, "val")
        self.plot_recon(pred, acts, "val")
        return pred
