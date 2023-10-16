import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

from data_loading import get_beatmap_files
from data_loading_img import get_img_data_loader
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class OsuModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, image):
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]

        logits_mask = self.forward(image)

        pred = torch.flatten(logits_mask, start_dim=1)
        loss = self.loss_fn(pred, mask)

        self.log(stage + "_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
        return torch.optim.Adam(self.parameters(), lr=0.0001)


def main(args):
    # Create training dataset
    train_dataloader = get_img_data_loader(
            dataset_path=args.data_path,
            start=0,
            end=16291,
            look_back_time=5000,
            cycle_length=args.batch_size // 4,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            beatmap_files=get_beatmap_files("splits/train_split.pkl", args.data_path)
        )
    validation_dataloader = get_img_data_loader(
        dataset_path=args.data_path,
        start=0,
        end=16291,
        look_back_time=5000,
        cycle_length=1,
        batch_size=args.batch_size * 2,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        beatmap_files=get_beatmap_files("splits/validation_split.pkl", args.data_path)
    )

    # Build model
    model = OsuModel("Unet", "resnet34", in_channels=1, out_classes=1, activation="identity")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="valid_loss",
        mode="min",
        dirpath="saved_models/",
        filename="test-{epoch:02d}-{valid_loss:.2f}",
        every_n_train_steps=20000,
    )

    trainer = pl.Trainer(
        max_epochs=5,
        val_check_interval=10000,
        callbacks=[checkpoint_callback]
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
    args = parser.parse_args()
    main(args)