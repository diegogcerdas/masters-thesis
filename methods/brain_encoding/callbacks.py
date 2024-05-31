import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torcheval.metrics.functional import r2_score
from datasets.nsd.utils.nsd_utils import (get_roi_indices, parse_rois)


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


class WandbCorrCallback(pl.Callback):
    def __init__(self, locs, hemisphere, subjdir):
        super().__init__()
        self.locs = locs
        self.hemisphere = hemisphere
        self.subjdir = subjdir
        self.targets = []
        self.preds = []

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self.targets = np.concatenate(self.targets)
        self.preds = np.concatenate(self.preds)

        metric = []
        for i in range(self.preds.shape[1]):
            metric.append(np.corrcoef(self.preds[:, i], self.targets[:, i])[0, 1])
        metric = np.array(metric)

        indices = set()
        for roi in [
            "V2v", 
            "V2d", 
            "V3v", 
            "V3d", 
            "hV4",
            "floc-bodies",
            "floc-faces",
            "floc-places",
            "floc-words",
            "midventral",
            "midlateral",
            "midparietal",
            "ventral",
            "lateral",
            "parietal",
        ] + (["V1v", "V1d", "early"] if self.hemisphere == "right" else []):
            roi_names, roi_classes = parse_rois([roi])
            ind = get_roi_indices(self.subjdir, roi_names, roi_classes, self.hemisphere)
            indices.update(ind)
        indices = list(indices)

        f = plt.figure(figsize=(5, 5))
        plt.scatter(self.locs[indices, 0], self.locs[indices, 1], c=metric[indices], cmap="hot", alpha=0.5)
        plt.colorbar(label="Correlation")   
        plt.title(f"Average: {metric.mean():.2f}")      
        plt.tight_layout()
        trainer.logger.log_image(key="R", images=[f])
        plt.show()

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