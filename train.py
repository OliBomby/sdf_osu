import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loading import get_beatmap_files
from data_loading_img import get_img_data_loader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class OsuModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=0.0001, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        logits_mask = self.forward(batch[0])
        pred = torch.flatten(logits_mask, start_dim=1)
        loss = self.loss_fn(pred, batch[1])

        self.log(stage + " loss", loss, prog_bar=True)

        return {
            "loss": loss,
        }

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main(args):
    # Create training dataset
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
            beatmap_files=get_beatmap_files("new_splits/train_split.pkl", args.data_path),
        )
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
        beatmap_files=get_beatmap_files("new_splits/validation_split.pkl", args.data_path),
        cache_dataset=True,
    )

    # Build model
    model = OsuModel("Unet", "resnet34", in_channels=1, out_classes=1, activation="identity")

    checkpoint_callback = ModelCheckpoint(
        dirpath="saved_models",
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        filename="{step:07d}-{valid_loss:.2f}",
        every_n_train_steps=20000,
    )

    wandb_logger = WandbLogger(
        project="sdf-osu",
        log_model=False if args.offline else "all",
        offline=args.offline
    )

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        max_epochs=5,
        val_check_interval=10000,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=args.log_every,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
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
    args = parser.parse_args()
    main(args)