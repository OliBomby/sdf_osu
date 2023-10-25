import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import WandbLogger
from torch import nn as nn

from constants import image_shape
from metrics import circle_accuracy, histogram_plot
from models import MitUnet


class OsuModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=0.0001, **kwargs):
        super().__init__()

        if arch == "Unet" and encoder_name.startswith("mit_") and in_channels != 3:
            self.model = MitUnet(encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        else:
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_test_step(self, batch, stage, batch_idx):
        logits_mask = self.forward(batch[0])
        pred = torch.flatten(logits_mask, start_dim=1)
        softmax_pred = torch.softmax(pred, dim=1)
        loss = self.loss_fn(pred, batch[1])
        metric = circle_accuracy(softmax_pred, batch[1])

        self.log(stage + "_loss", loss, prog_bar=True)
        self.log(stage + "_circle_accuracy", metric, prog_bar=True)

        if isinstance(self.logger, WandbLogger) and batch_idx == 0:
            num_img = 16
            colormap = plt.get_cmap('viridis')
            prior_images = colormap(batch[0][:num_img].squeeze(1).cpu())
            prediction_images = colormap(torch.pow(softmax_pred[:num_img].reshape((-1,) + image_shape), 1/4).cpu())
            combined_images = np.concatenate((prior_images, prediction_images), axis=2)
            split_images = list(np.split(combined_images, num_img, axis=0))
            # noinspection PyUnresolvedReferences
            self.logger.log_image(key=stage + "_samples", images=split_images)

        result = {
            "loss": loss,
            "circle_accuracy": metric
        }

        if stage == "test":
            # Calculate visual spacing distribution
            histogram = histogram_plot(softmax_pred, batch[0], 0.1, batch[2], max_distance=4, min_value=0.6)
            result["histogram"] = histogram

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_load_checkpoint(self, checkpoint):
        checkpoint["optimizer_states"][0]["param_groups"][0]['capturable'] = True
