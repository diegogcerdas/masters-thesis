import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from torcheval.metrics.functional import r2_score


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        feature_extractor,
        input_size: int,
        output_size: int,
        learning_rate: float,
        brain_wise_mean: float,
        brain_wise_std: float,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters(ignore="feature_extractor")
        self.feature_extractor = feature_extractor
        self.learning_rate = learning_rate
        self.brain_wise_mean = brain_wise_mean
        self.brain_wise_std = brain_wise_std

        self.encoder = torch.nn.Linear(input_size, output_size)

    def forward(self, x, mode="val", no_grad=True):
        if no_grad:
            with torch.no_grad():
                x = self.feature_extractor(x, mode)
        else:
            x = self.feature_extractor(x, mode)
        x = self.encoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
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
