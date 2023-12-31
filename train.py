import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data_loading import load_splits, get_cached_data_loader
from data_loading_img import get_img_data_loader
import pytorch_lightning as pl

from lightning_model import OsuModel

# Faster, but less precise
torch.set_float32_matmul_precision("high")


def main(args):
    # Build model
    model = OsuModel(args.arch, args.encoder_name, in_channels=1, out_classes=1, activation="identity", encoder_weights=args.encoder_weights)

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

    if args.cached_train_data is None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="saved_models",
            save_top_k=2,
            monitor="valid_loss",
            mode="min",
            filename="{step:07d}-{valid_loss:.2f}",
            every_n_train_steps=2000,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath="saved_models",
            save_top_k=2,
            monitor="valid_loss",
            mode="min",
            filename="{epoch:02d}-{valid_loss:.2f}",
            every_n_epochs=1,
        )

    wandb_logger = WandbLogger(
        project="sdf-osu",
        name=f"{model.encoder_name} {model.arch} big",
        log_model=False if args.offline else "all",
        offline=args.offline
    )

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    trainer = pl.Trainer(
        max_epochs=-1,
        val_check_interval=1000 if args.cached_train_data is None else None,
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
    parser.add_argument("--arch", type=str, default="Unet")
    parser.add_argument("--encoder-name", type=str, default="mit_b0")
    parser.add_argument("--encoder-weights", type=str, default=None)
    args = parser.parse_args()
    main(args)
