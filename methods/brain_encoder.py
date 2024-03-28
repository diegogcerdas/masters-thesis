import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import r2_score
from methods.feature_extractor import create_feature_extractor


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        feature_extractor_type: str,
        output_size: int,
        learning_rate: float,
        device: str,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters()
        self.feature_extractor = create_feature_extractor(feature_extractor_type, device)
        self.feature_extractor.requires_grad_(False)
        self.learning_rate = learning_rate
        self.encoder = torch.nn.Linear(self.feature_extractor.feature_size, output_size)

    def forward(self, x, mode="val", no_grad=True):
        if no_grad:
            with torch.no_grad():
                x = self.feature_extractor(x, mode)
        else:
            x = self.feature_extractor(x, mode)
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch, mode):
        img, target, _ = batch
        pred = self(img, mode).squeeze()
        loss = F.mse_loss(pred, target)
        metric = r2_score(pred, target)
        self.log_stat(f"{mode}_r2", metric)
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