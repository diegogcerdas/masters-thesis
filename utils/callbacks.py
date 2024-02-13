import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torcheval.metrics.functional import r2_score


class WandbTSNECallback(pl.Callback):
    def __init__(self, predict_average: bool):
        super().__init__()
        self.predict_average = predict_average
        self.targets = []
        self.preds = []
        self.low_dim = []

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.targets = np.array(self.targets)
        self.preds = np.array(self.preds)
        self.low_dim = np.concatenate(self.low_dim)

        f = plt.figure(figsize=(6, 5))
        plt.scatter(
            self.low_dim[:, 0],
            self.low_dim[:, 1],
            c=self.targets,
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            s=5,
        )
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        trainer.logger.log_image(key="Targets", images=[f])
        plt.close()

        f = plt.figure(figsize=(6, 5))
        plt.scatter(
            self.low_dim[:, 0],
            self.low_dim[:, 1],
            c=self.preds,
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            s=5,
        )
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        trainer.logger.log_image(key="Predictions", images=[f])
        plt.close()

        self.targets = []
        self.preds = []
        self.low_dim = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        _, target, low_dim = batch
        pred = outputs
        if not self.predict_average:
            target = target.mean(dim=1)
            pred = pred.mean(dim=1)
        target = target.squeeze().detach().cpu().numpy().tolist()
        low_dim = low_dim.squeeze().detach().cpu().numpy()
        pred = pred.squeeze().detach().cpu().numpy().tolist()
        self.targets = self.targets + target
        self.preds = self.preds + pred
        self.low_dim.append(low_dim)


class WandbR2Callback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.preds = []

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.targets = np.concatenate(self.targets)
        self.preds = np.concatenate(self.preds)
        metric = r2_score(
            torch.tensor(self.preds),
            torch.tensor(self.targets),
            multioutput="raw_values",
        )

        f = plt.figure(figsize=(15, 5))
        plt.bar(range(len(metric)), metric)
        plt.tight_layout()
        trainer.logger.log_image(key="R2", images=[f])
        plt.close()

        self.targets = []
        self.preds = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        _, target, _ = batch
        target = target.squeeze().detach().cpu().numpy()
        pred = outputs.squeeze().detach().cpu().numpy()
        self.targets.append(target)
        self.preds.append(pred)
