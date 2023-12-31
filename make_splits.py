import json
import os
import pickle
import random


def _get_beatmap_files(self) -> list[str]:
    # Get a list of all beatmap files in the dataset path in the track index range between start and end
    beatmap_files = []
    track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
    for track_name in track_names:
        if self.subset_ids is not None:
            metadata_File = os.path.join(
                self.dataset_path,
                track_name,
                "metadata.json",
            )
            with open(metadata_File) as f:
                metadata = json.load(f)
            for beatmap_name in metadata["Beatmaps"]:
                beatmap_metadata = metadata["Beatmaps"][beatmap_name]
                if beatmap_metadata["BeatmapId"] in self.subset_ids:
                    beatmap_files.append(
                        os.path.join(
                            self.dataset_path,
                            track_name,
                            "beatmaps",
                            beatmap_name + ".osu",
                        ),
                    )
        else:
            for beatmap_file in os.listdir(
                    os.path.join(self.dataset_path, track_name, "beatmaps"),
            ):
                beatmap_files.append(
                    os.path.join(
                        self.dataset_path,
                        track_name,
                        "beatmaps",
                        beatmap_file,
                    ),
                )

    return beatmap_files


def get_beatmap_files(data_path: str, mapset_indices: list[int], min_sr: float = 0) -> list[str]:
    # Get a list of all beatmap files in the dataset path in the track index range between start and end
    beatmap_files = []
    for i in mapset_indices:
        track_name = "Track" + str(i).zfill(5)
        if min_sr > 0:
            metadata_File = os.path.join(
                data_path,
                track_name,
                "metadata.json",
            )
            with open(metadata_File) as f:
                metadata = json.load(f)
            for beatmap_name in metadata["Beatmaps"]:
                beatmap_metadata = metadata["Beatmaps"][beatmap_name]
                if beatmap_metadata["StandardStarRating"]["0"] >= min_sr:
                    beatmap_files.append(
                        os.path.join(
                            track_name,
                            "beatmaps",
                            beatmap_name + ".osu",
                        ),
                    )
        else:
            for beatmap_file in os.listdir(
                    os.path.join(data_path, track_name, "beatmaps"),
            ):
                beatmap_file_relative_path = os.path.join(
                    track_name,
                    "beatmaps",
                    beatmap_file,
                )
                beatmap_files.append(beatmap_file_relative_path)

    return beatmap_files


def main(args):
    all_train_set = list(range(args.train_start, args.train_end))
    all_train_files = get_beatmap_files(args.data_path, all_train_set, args.min_sr)

    validation_files = random.sample(all_train_files, args.validation_count)
    train_files = [f for f in all_train_files if f not in validation_files]

    test_set = list(range(args.test_start, args.test_end))
    test_files = get_beatmap_files(args.data_path, test_set, args.min_sr)

    if args.train_count is not None:
        train_files = random.sample(train_files, args.train_count)

    if args.test_count is not None:
        test_files = random.sample(test_files, args.test_count)

    # Print the lengths of the lists
    print(f"Length of train_split: {len(train_files)}")
    print(f"Length of validation_split: {len(validation_files)}")
    print(f"Length of test_split: {len(test_files)}")

    # Create the 'splits' folder
    splits_folder = args.out_dir
    if not os.path.exists(splits_folder):
        os.makedirs(splits_folder)

    # Write the lists to separate files using pickle
    with open(os.path.join(splits_folder, 'train_split.pkl'), 'wb') as train_file:
        pickle.dump(train_files, train_file)

    with open(os.path.join(splits_folder, 'validation_split.pkl'), 'wb') as validation_file:
        pickle.dump(validation_files, validation_file)

    with open(os.path.join(splits_folder, 'test_split.pkl'), 'wb') as test_file:
        pickle.dump(test_files, test_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--train-start", type=int, required=True)
    parser.add_argument("--train-end", type=int, required=True)
    parser.add_argument("--test-start", type=int, required=True)
    parser.add_argument("--test-end", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--test-count", type=int, default=None)
    parser.add_argument("--min-sr", type=float, default=0)
    parser.add_argument("--out-dir", type=str, default="splits")
    args = parser.parse_args()
    main(args)
