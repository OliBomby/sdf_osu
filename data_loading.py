import math
import os.path
import pickle
import random
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path, PureWindowsPath
from typing import Optional

import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, IterableDataset

from slider import Position
from slider.beatmap import Beatmap
from slider.beatmap import HitObject
from slider.beatmap import Slider
from slider.beatmap import Spinner
from slider.curve import Catmull
from slider.curve import Linear
from slider.curve import MultiBezier
from slider.curve import Perfect

playfield_size = torch.tensor((512, 384))
feature_size = 12


def create_datapoint(time: timedelta, pos: Position, datatype: int) -> torch.Tensor:
    features = torch.zeros(feature_size)
    features[0] = pos.x
    features[1] = pos.y
    features[2] = time.total_seconds() * 1000
    features[datatype + 3] = 1

    return features


def repeat_type(repeat: int) -> int:
    if repeat % 2 == 0:
        return 0
    else:
        return 1


def append_control_points(
        datapoints: list[torch.Tensor],
        slider: Slider,
        datatype: int,
        duration: timedelta,
):
    control_point_count = len(slider.curve.points)

    for i in range(1, control_point_count - 1):
        time = slider.time + i / (control_point_count - 1) * duration
        pos = slider.curve.points[i]
        datapoints.append(create_datapoint(time, pos, datatype))


def get_data(hitobj: HitObject) -> torch.Tensor:
    if isinstance(hitobj, Slider) and len(hitobj.curve.points) < 100:
        datapoints = [
            create_datapoint(
                hitobj.time,
                hitobj.position,
                1,
            ),
        ]

        assert hitobj.repeat >= 1
        duration: timedelta = (hitobj.end_time - hitobj.time) / hitobj.repeat

        if isinstance(hitobj.curve, Linear):
            append_control_points(datapoints, hitobj, 5, duration)
        elif isinstance(hitobj.curve, Catmull):
            append_control_points(datapoints, hitobj, 4, duration)
        elif isinstance(hitobj.curve, Perfect):
            append_control_points(datapoints, hitobj, 3, duration)
        elif isinstance(hitobj.curve, MultiBezier):
            control_point_count = len(hitobj.curve.points)

            for i in range(1, control_point_count - 1):
                time = hitobj.time + i / (control_point_count - 1) * duration
                pos = hitobj.curve.points[i]

                if pos == hitobj.curve.points[i + 1]:
                    datapoints.append(create_datapoint(time, pos, 5))
                elif pos != hitobj.curve.points[i - 1]:
                    datapoints.append(create_datapoint(time, pos, 2))

        datapoints.append(
            create_datapoint(hitobj.time + duration, hitobj.curve.points[-1], 6),
        )

        slider_end_pos = hitobj.curve(1)
        datapoints.append(
            create_datapoint(
                hitobj.end_time,
                slider_end_pos,
                7 + repeat_type(hitobj.repeat),
            ),
        )

        return torch.stack(datapoints, 0)

    if isinstance(hitobj, Spinner):
        raise Exception("Spinners shouldn't exist")

    return create_datapoint(
        hitobj.time,
        hitobj.position,
        0,
    ).unsqueeze(0)


def beatmap_to_sequence(beatmap: Beatmap) -> torch.Tensor:
    # Get the hit objects
    hit_objects = beatmap.hit_objects(stacking=False, spinners=False)
    data_chunks = [get_data(ho) for ho in hit_objects]

    sequence = torch.concatenate(data_chunks, 0)

    return sequence.float()


def random_flip(seq: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        seq[:, 0] = 512 - seq[:, 0]
    if random.random() < 0.5:
        seq[:, 1] = 384 - seq[:, 1]
    return seq


def calc_distances(seq: torch.Tensor) -> torch.Tensor:
    offset = torch.roll(seq[:, :2], 1, 1)
    offset[0, 0] = 256
    offset[0, 1] = 192
    seq_d = torch.linalg.vector_norm(seq[:, :2] - offset, ord=2, dim=1)
    return seq_d


def split_and_process_sequence(
        seq: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    # Augment positions with random flip
    # seq_aug = random_flip(seq)

    return seq, seq.shape[0]


def load_and_process_beatmap(beatmap: Beatmap):
    seq = beatmap_to_sequence(beatmap)
    return split_and_process_sequence(seq)


def window_and_relative_time(seq, s, e):
    window = seq[s:e]
    # Obscure the absolute time by normalizing to zero
    window[:, 2] -= window[0, 2]

    return window


class BeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "beatmap_idx",
        "seq_len",
        "stride",
        "index",
        "current_idx",
        "current_seq",
        "current_seq_len",
        "seq_index",
        "seq_func",
        "win_func",
    )

    def __init__(
            self,
            beatmap_files: list[str],
            seq_len: int,
            stride: int,
            seq_func: Optional[Callable] = None,
            win_func: Optional[Callable] = None,
    ):
        self.beatmap_files = beatmap_files
        self.seq_len = seq_len
        self.stride = stride
        self.index = 0
        self.current_idx = 0
        self.current_seq = None
        self.current_seq_len = -1
        self.seq_index = 0
        self.seq_func = (
            seq_func if seq_func is not None else lambda x: beatmap_to_sequence(x)
        )
        self.win_func = win_func if win_func is not None else lambda x, s, e: x[:, s:e]

    def __iter__(self) -> "BeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        while (
                self.current_seq is None
                or self.seq_index + self.seq_len > self.current_seq_len
        ):
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap = Beatmap.from_path(beatmap_path)

            self.current_idx = int(os.path.basename(beatmap_path)[:6])
            self.current_seq, self.current_seq_len = self.seq_func(beatmap)
            self.seq_index = random.randint(0, self.stride - 1)
            self.index += 1

        # Return the preprocessed hit objects as a sequence of overlapping windows
        window = self.win_func(
            self.current_seq,
            self.seq_index,
            self.seq_index + self.seq_len,
        )
        self.seq_index += self.stride
        return window, self.current_idx


class InterleavingBeatmapDatasetIterable:
    __slots__ = ("workers", "cycle_length", "index")

    def __init__(
            self,
            beatmap_files: list[str],
            iterable_factory: Callable,
            cycle_length: int,
    ):
        per_worker = int(math.ceil(len(beatmap_files) / float(cycle_length)))
        self.workers = [
            iterable_factory(
                beatmap_files[
                    i * per_worker: min(len(beatmap_files), (i + 1) * per_worker)
                ]
            )
            for i in range(cycle_length)
        ]
        self.cycle_length = cycle_length
        self.index = 0

    def __iter__(self) -> "InterleavingBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[any, int]:
        num = len(self.workers)
        for _ in range(num):
            try:
                self.index = self.index % len(self.workers)
                item = self.workers[self.index].__next__()
                self.index += 1
                return item
            except StopIteration:
                self.workers.remove(self.workers[self.index])
        raise StopIteration


class BeatmapDataset(IterableDataset):
    def __init__(
            self,
            dataset_path: str,
            start: int,
            end: int,
            iterable_factory: Callable,
            cycle_length: int = 1,
            shuffle: bool = False,
            beatmap_files: Optional[list[str]] = None,
    ):
        super(BeatmapDataset).__init__()
        self.dataset_path = dataset_path
        self.start = start
        self.end = end
        self.iterable_factory = iterable_factory
        self.cycle_length = cycle_length
        self.shuffle = shuffle
        self.beatmap_files = beatmap_files

    def _get_beatmap_files(self) -> list[str]:
        if self.beatmap_files is not None:
            return self.beatmap_files

        # Get a list of all beatmap files in the dataset path in the track index range between start and end
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(self.start, self.end)]
        for track_name in track_names:
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

    def __iter__(self) -> InterleavingBeatmapDatasetIterable | BeatmapDatasetIterable:
        beatmap_files = self._get_beatmap_files()

        if self.shuffle:
            random.shuffle(beatmap_files)

        if self.cycle_length > 1:
            return InterleavingBeatmapDatasetIterable(
                beatmap_files,
                self.iterable_factory,
                self.cycle_length,
            )

        return self.iterable_factory(beatmap_files)


# Define a `worker_init_fn` that configures each dataset copy differently
def worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(
        math.ceil((overall_end - overall_start) / float(worker_info.num_workers)),
    )
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


def get_beatmap_files(name: str, data_path: str) -> list[str]:
    p = Path(name)
    with p.open("rb") as f:
        relative_beatmap_files = pickle.load(f)
    beatmap_files = [os.path.join(data_path, PureWindowsPath(f)) for f in relative_beatmap_files]
    return beatmap_files


class BeatmapDatasetIterableFactory:
    __slots__ = ("seq_len", "stride", "seq_func", "win_func")

    def __init__(self, seq_len, stride, seq_func, win_func):
        self.seq_len = seq_len
        self.stride = stride
        self.seq_func = seq_func
        self.win_func = win_func

    def __call__(self, *args, **kwargs):
        beatmap_files = args[0]
        return BeatmapDatasetIterable(
            beatmap_files=beatmap_files,
            seq_len=self.seq_len,
            stride=self.stride,
            seq_func=self.seq_func,
            win_func=self.win_func,
        )


def get_tabular_data_loader(
        dataset_path: str,
        start: int,
        end: int,
        seq_len: int,
        stride: int = 1,
        cycle_length=1,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        beatmap_files: Optional[list[str]] = None,
        cache_dataset: bool = False,
) -> DataLoader:
    return get_data_loader(
        dataset_path=dataset_path,
        start=start,
        end=end,
        iterable_factory=BeatmapDatasetIterableFactory(
            seq_len,
            stride,
            load_and_process_beatmap,
            window_and_relative_time,
        ),
        cycle_length=cycle_length,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        beatmap_files=beatmap_files,
        cache_dataset=cache_dataset,
    )


class CachedDataset(Dataset):
    __slots__ = "cached_data"

    def __init__(self, cached_data):
        self.cached_data = cached_data

    def __getitem__(self, index):
        return self.cached_data[index]

    def __len__(self):
        return len(self.cached_data)


def cache_dataset(
        out_path: str,
        dataset_path: str,
        start: int,
        end: int,
        iterable_factory: Callable,
        cycle_length=1,
        beatmap_files: Optional[list[str]] = None,
):
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        start=start,
        end=end,
        iterable_factory=iterable_factory,
        cycle_length=cycle_length,
        shuffle=False,
        beatmap_files=beatmap_files,
    )

    print("Caching dataset...")
    cached_data = []
    for datum in tqdm.tqdm(dataset):
        cached_data.append(datum)

    torch.save(cached_data, out_path)


def get_cached_data_loader(
        data_path: str,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
):
    cached_data = torch.load(data_path)
    dataset = CachedDataset(cached_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
    )

    return dataloader


def get_data_loader(
        dataset_path: str,
        start: int,
        end: int,
        iterable_factory: Callable,
        cycle_length=1,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        beatmap_files: Optional[list[str]] = None,
) -> DataLoader:
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        start=start,
        end=end,
        iterable_factory=iterable_factory,
        cycle_length=cycle_length,
        shuffle=shuffle,
        beatmap_files=beatmap_files,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def load_splits(splits_dir, data_path):
    train_split = None
    validation_split = None
    test_split = None
    if splits_dir is not None:
        train_split = get_beatmap_files(os.path.join(splits_dir, "train_split.pkl"), data_path)
        validation_split = get_beatmap_files(os.path.join(splits_dir, "validation_split.pkl"), data_path)
        test_split = get_beatmap_files(os.path.join(splits_dir, "test_split.pkl"), data_path)

    return train_split, validation_split, test_split


def main(args):
    dataloader = get_tabular_data_loader(
        dataset_path=args.data_path,
        start=0,
        end=16291,
        seq_len=128,
        stride=16,
        cycle_length=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    if args.mode == "plotfirst":
        import matplotlib.pyplot as plt

        for x, y in dataloader:
            print(x.shape, y.shape)
            print(x[0, 0])
            print(y)

            # Normalize for visualization purposes
            x[:, :, 0] /= 512
            x[:, :, 1] /= 384
            x[:, :, 2] /= torch.max(x[:, :, 2])

            for j in range(args.batch_size):
                fig, axs = plt.subplots(1, figsize=(5, 20))
                axs.imshow(x[j])
                print(y[j])
                plt.show()
            break
    elif args.mode == "benchmark":
        import tqdm
        for _ in tqdm.tqdm(dataloader, total=7000, smoothing=0.01):
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    main(args)
