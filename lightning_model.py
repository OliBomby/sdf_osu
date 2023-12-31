from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import nn as nn

from constants import image_shape
from data_loading_img import coord_index_to_coord
from lib.models.model_manager import ModelManager
from lib.utils.tools.configer import Configer
from metrics import circle_accuracy, ds_histogram
from models import MitUnet


class OsuModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=0.0001, **kwargs):
        super().__init__()

        if arch == "Unet" and encoder_name.startswith("mit_") and in_channels != 3:
            self.model = MitUnet(encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        elif arch.startswith("hrt"):
            configer = Configer(config_dict={
                "data": {
                    "num_classes": 1,
                },
                "network": {
                    "model_name": arch,
                    "backbone": encoder_name,
                    "pretrained": None,
                    "multi_grid": [1, 1, 1],
                    "bn_type": "torchsyncbn",
                    "stride": 8,
                    "factors": [[8, 8]],
                    "loss_weights": {
                        "corr_loss": 0.01,
                        "aux_loss": 0.4,
                        "seg_loss": 1.0
                    }
                },
            })
            self.model = ModelManager(configer).semantic_segmentor()
        else:
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )
        self.arch = arch
        self.encoder_name = encoder_name
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

        self.test_ds_histograms = []
        self.test_name = None
        self.log_all_images = False

    def forward(self, image):
        mask = self.model(image)
        if isinstance(mask, Tuple):
            # Ignore the auxiliary output
            mask = mask[1]
        return mask

    def shared_test_step(self, batch, stage, batch_idx):
        logits_mask = self.forward(batch[0])
        pred = torch.flatten(logits_mask, start_dim=1)
        softmax_pred = torch.softmax(pred, dim=1)
        # softmax_pred = torch.nn.functional.one_hot(batch[1], flat_num).float()
        loss = self.loss_fn(pred, batch[1])
        # loss = 0
        metric = circle_accuracy(softmax_pred, batch[1])

        test_name = "_" + self.test_name if self.test_name is not None else ""
        self.log(stage + test_name + "_loss", loss, prog_bar=True)
        self.log(stage + test_name + "_circle_accuracy", metric, prog_bar=True)

        if isinstance(self.logger, WandbLogger) and (batch_idx == 0 or self.log_all_images):
            num_img = min(16, batch[0].shape[0])
            colormap = plt.get_cmap('viridis')
            prior_images = colormap(batch[0][:num_img].squeeze(1).cpu())
            prediction_images = colormap(torch.pow(softmax_pred[:num_img].reshape((-1,) + image_shape), 1/4).cpu())
            true_x, true_y = coord_index_to_coord(batch[1][:num_img].cpu())
            prior_images[:, true_y, true_x] = (1, 0, 0, 1)
            prediction_images[:, true_y, true_x] = (1, 0, 0, 1)
            combined_images = np.concatenate((prior_images, prediction_images), axis=2)
            split_images = list(np.split(combined_images, num_img, axis=0))
            # noinspection PyUnresolvedReferences
            self.logger.log_image(key=stage + test_name + "_samples", images=split_images)

        result = {
            "loss": loss,
            "circle_accuracy": metric
        }

        if stage == "test":
            # Calculate visual spacing distribution
            histogram = ds_histogram(softmax_pred, batch[0], 0.1, batch[2], max_distance=8, min_value=0.6)
            self.test_ds_histograms.append(histogram)

        return result

    def training_step(self, batch, batch_idx):
        logits_mask = self.forward(batch[0])
        pred = torch.flatten(logits_mask, start_dim=1)
        loss = self.loss_fn(pred, batch[1])

        self.log("train_loss", loss, prog_bar=True)

        return {
            "loss": loss,
        }

    def validation_step(self, batch, batch_idx):
        return self.shared_test_step(batch, "valid", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_test_step(batch, "test", batch_idx)

    def on_test_epoch_end(self):
        if len(self.test_ds_histograms) == 0:
            return

        # we want to aggregate the histograms. We'll start with the first one and then add each other one to it.
        ds_histogram = np.sum(np.stack(self.test_ds_histograms, axis=0), axis=0)  # start with the first histogram
        self.test_ds_histograms.clear()

        # Normalize the histogram
        total_count = np.sum(ds_histogram)  # sum of all bin counts

        # To avoid division by zero, check if total_count is zero
        if total_count > 0:
            ds_histogram = ds_histogram / total_count  # normalize each bin count

        # WANDB LOGGING
        if isinstance(self.logger, WandbLogger):
            # Create a range for the x-axis (bins)
            xs = np.arange(0, 8, 0.1).tolist()

            # Prepare your data for logging
            ys = [ds_histogram.tolist()]  # Make sure it's a list of lists

            # Create a custom wandb plot without directly importing wandb
            line_plot = wandb.plot.line_series(
                xs=xs,  # Your x-axis data (bins)
                ys=ys,  # Your y-axis data (normalized histogram counts)
                title="Visual spacing distribution",
                xname="Distance",
                keys=["Density"],
            )

            # Log the custom plot using the logger's experiment attribute
            # noinspection PyUnresolvedReferences
            self.logger.experiment.log({"ds_histogram": line_plot})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_load_checkpoint(self, checkpoint):
        if torch.__version__.startswith("1.12"):
            checkpoint["optimizer_states"][0]["param_groups"][0]['capturable'] = True
