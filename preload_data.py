import os

from data_loading import cache_dataset, load_splits
from data_loading_img import ImgBeatmapDatasetIterableFactory


def main(args):
    train_split, validation_split, test_split = load_splits(args.splits_dir, args.data_path)

    def cache_data(split, filename):
        cache_dataset(
            out_path=os.path.join(args.out_dir, filename),
            dataset_path=args.data_path,
            start=0,
            end=16291,
            iterable_factory=ImgBeatmapDatasetIterableFactory(
                look_back_time=5000,
            ),
            cycle_length=1,
            beatmap_files=split,
        )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    cache_data(train_split, "train_data.pt")
    cache_data(validation_split, "validation_data.pt")
    cache_data(test_split, "test_data.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--splits-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)