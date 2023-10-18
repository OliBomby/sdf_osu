import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loading import load_splits, get_cached_data_loader
from data_loading_img import get_img_data_loader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from metrics import circle_accuracy
from models import MitUnet
from constants import image_shape

# Faster, but less precise
torch.set_float32_matmul_precision("high")


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

        return {
            "loss": loss,
            "circle_accuracy": metric
        }

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


def main(args):
    # Build model
    model = OsuModel("Unet", "mit_b0", in_channels=1, out_classes=1, activation="identity", encoder_weights=None)

    # Load splits
    train_split, validation_split, test_split = load_splits(args.splits_dir, args.data_path)

    # Create training dataset
    if args.cached_train_data is not None:
        train_dataloader = get_cached_data_loader(
            data_path=args.cached_train_data,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_dataloader = get_img_data_loader(
            dataset_path=args.data_path,
            start=0,
            end=16291,
            look_back_time=5000,
            cycle_length=args.batch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            beatmap_files=train_split,
        )

    if args.cached_val_data is not None:
        validation_dataloader = get_cached_data_loader(
            data_path=args.cached_val_data,
            batch_size=args.batch_size * 2,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    else:
        validation_dataloader = get_img_data_loader(
            dataset_path=args.data_path,
            start=0,
            end=16291,
            look_back_time=5000,
            cycle_length=1,
            batch_size=args.batch_size * 2,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            beatmap_files=validation_split,
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved_models",
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        # filename="{epoch:02d}-{valid_loss:.2f}",
        # every_n_epochs=1,
        filename="{step:07d}-{valid_loss:.2f}",
        every_n_train_steps=2000,
    )

    wandb_logger = WandbLogger(
        project="sdf-osu",
        log_model=False if args.offline else "all",
        offline=args.offline
    )

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        max_epochs=-1,
        val_check_interval=1000,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=args.log_every,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        ckpt_path=args.ckpt,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--offline", type=bool, default=False)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--cached-train-data", type=str, default=None)
    parser.add_argument("--cached-val-data", type=str, default=None)
    args = parser.parse_args()
    main(args)
