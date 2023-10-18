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

from models import MitUnet
from constants import image_shape
from metrics import metric1

from train import OsuModel


def main(args):
    # Build model
    model = pl.LightningModule.load_from_checkpoint(args.ckpt)

    # Load splits
    _, _, test_split = load_splits(args.splits_dir, args.data_path)

    if args.cached_test_data is not None:
        test_dataloader = get_cached_data_loader(
            data_path=args.cached_val_data,
            batch_size=args.batch_size * 2,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    else:
        test_dataloader = get_img_data_loader(
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
            beatmap_files=test_split,
            cache_dataset=True,
        )

    wandb_logger = WandbLogger(
        project="sdf-osu",
        log_model=False if args.offline else "all",
        offline=args.offline
    )

    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log="all")

    pl.Trainer.test(
        model,
        dataloaders=test_dataloader,
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
    parser.add_argument("--cached-test-data", type=str, default=None)
    args = parser.parse_args()
    main(args)
