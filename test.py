import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from data_loading import load_splits, get_cached_data_loader
from data_loading_img import get_img_data_loader
from lightning_model import OsuModel


def main(args):
    # Build model
    model = OsuModel.load_from_checkpoint(args.ckpt)

    # Load splits
    _, validation_split, test_split = load_splits(args.splits_dir, args.data_path)

    if args.cached_test_data is not None:
        test_dataloader = get_cached_data_loader(
            data_path=args.cached_test_data,
            batch_size=args.batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    else:
        test_dataloader = get_img_data_loader(
            dataset_path=args.data_path,
            start=0,
            end=args.data_end,
            look_back_time=5000,
            cycle_length=1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            beatmap_files=validation_split if args.validate else test_split,
        )

    test_name = "validation" if args.validate else "test"

    wandb_logger = WandbLogger(
        project="sdf-osu",
        name=f"{test_name} {model.encoder_name} {model.arch}",
        offline=args.offline,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
    )

    if args.validate:
        trainer.validate(
            model,
            dataloaders=test_dataloader,
        )
    else:
        trainer.test(
            model,
            dataloaders=test_dataloader,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-end", type=int, default=16291)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--offline", type=bool, default=False)
    parser.add_argument("--splits-dir", type=str, default=None)
    parser.add_argument("--cached-test-data", type=str, default=None)
    parser.add_argument("--validate", type=bool, default=False)
    args = parser.parse_args()
    main(args)
