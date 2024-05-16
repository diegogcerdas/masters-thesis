import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torcheval.metrics.functional import r2_score
from methods.high_level_attributes.clip_extractor import create_feature_extractor

def create_encoder_model(encoder_type, output_size, device):
    return


class EncoderModule(pl.LightningModule):
    def __init__(
        self,
        encoder_type: str,
        output_size: int,
        learning_rate: float,
        device: str,
    ):
        super(EncoderModule, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = create_encoder_model(encoder_type, output_size, device)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def compute_loss(self, batch, mode):
        img, target, _ = batch
        pred = self(img).squeeze()
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