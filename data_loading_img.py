import math
import time
from typing import Optional

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from slider.beatmap import Beatmap, HitObject, Slider
from constants import coordinates, playfield_width_num, playfield_height_num, \
    max_sdf_distance, image_shape
from data_loading import get_data_loader
from plotting import plot_signed_distance_field

empty_pos_tensor = torch.zeros((0, 2), dtype=torch.float32)
empty_sdf_array = torch.full(image_shape, max_sdf_distance, dtype=torch.float32)


def get_hit_object_radius(circle_size):
    return (109 - 9 * circle_size) / 2


def geometry_to_sdf3(geometry, radius):
    """
    Calculates the playfield SDF from a list of positions representing the geometry, normalized to the radius
    :param geometry: Tensor of shape (M, 2) representing the positions
    :param radius: Scalar representing the radius of the geometry
    :return: The SDF
    """
    return np.minimum(
        np.sqrt(
            np.min(
                np.sum(
                    np.square(
                        np.tile(
                            np.reshape(
                                geometry,
                                (1, 1, -1, 2)
                            ),
                            (playfield_height_num, playfield_width_num, 1, 1)
                        ) - coordinates),
                    axis=-1
                ),
                axis=-1
            )
        ) / radius - 1,
        max_sdf_distance
    )


def get_coord_index3(pos: Tensor):
    x_index = int(np.clip(np.round(pos[0] / 4), 0, playfield_width_num - 1))
    y_index = int(np.clip(np.round(pos[1] / 4), 0, playfield_height_num - 1))
    return y_index * playfield_width_num + x_index


def trajectory_to_img(trajectory, next_time, look_back_time):
    img = torch.zeros((playfield_height_num, playfield_width_num), dtype=torch.float32)
    return draw_trajectory(img, trajectory, next_time, look_back_time)


def draw_trajectory(img, trajectory, next_time, look_back_time):
    x_indices = torch.clip(torch.round(trajectory[:, 0] / 4), 0, playfield_width_num - 1).long()
    y_indices = torch.clip(torch.round(trajectory[:, 1] / 4), 0, playfield_height_num - 1).long()
    img[y_indices, x_indices] = 1 - (next_time - trajectory[:, 2]) / look_back_time
    return img


def get_trajectory(ho: HitObject):
    if isinstance(ho, Slider) and len(ho.curve.points) < 200:
        repeat = ho.repeat
        dur = ho.end_time - ho.time
        repeat_dur = dur / repeat
        last_repeat_start = ho.time + repeat_dur * (repeat - 1)

        if repeat % 2 == 0:
            # Slider ends at the start position
            return [ho.curve(1 - t) + ((last_repeat_start + t * repeat_dur).total_seconds() * 1000,) for t in np.linspace(0, 1, 100)]
        else:
            return [ho.curve(t) + ((last_repeat_start + t * repeat_dur).total_seconds() * 1000,) for t in np.linspace(0, 1, 100)]

    return [ho.position + (ho.time.total_seconds() * 1000,)]


def get_timestep_embedding(timestep, dim=64, max_period=10000):
    half = dim // 2
    freqs = np.exp(
        -math.log(max_period) * np.arange(0, half, dtype=np.float32) / half
    )
    args = np.asarray([timestep], dtype=np.float32) * freqs
    embedding = np.concatenate([np.cos(args), np.sin(args)], 0)
    return embedding


def buffers_to_sdf_tensor(buffers):
    return np.stack(
        [(np.min(np.stack([data[1] for data in buffer], axis=-1), axis=-1) if len(buffer) > 0 else empty_sdf_array) for
         buffer in buffers], axis=-1)


class ImgBeatmapDatasetIterable:
    __slots__ = (
        "beatmap_files",
        "beatmap_idx",
        "look_back_time",
        "index",
        "radius",
        "hit_objects",
        "ho_index",
        "look_back_time",
        "prev_img",
        "prev_trajectory",
        "circle_radius",
    )

    def __init__(
            self,
            beatmap_files: list[str],
            look_back_time: float,
    ):
        self.beatmap_files = beatmap_files
        self.look_back_time = look_back_time
        self.index = 0
        self.radius = 1
        self.hit_objects = None
        self.ho_index = 0
        self.prev_img = None
        self.prev_trajectory = None
        self.circle_radius = 1

    def __iter__(self) -> "ImgBeatmapDatasetIterable":
        return self

    def __next__(self) -> tuple[Tensor, Tensor, float]:
        while (
                self.hit_objects is None
                or self.ho_index >= len(self.hit_objects)
        ):
            if self.index >= len(self.beatmap_files):
                raise StopIteration

            # Load the beatmap from file
            beatmap_path = self.beatmap_files[self.index]
            beatmap: Beatmap = Beatmap.from_path(beatmap_path)
            self.radius = get_hit_object_radius(beatmap.circle_size)
            self.hit_objects = beatmap.hit_objects(spinners=False)
            self.prev_trajectory = None
            self.prev_img = torch.zeros((playfield_height_num, playfield_width_num), dtype=torch.float32)
            self.ho_index = 0
            self.circle_radius = get_hit_object_radius(beatmap.circle_size)
            self.index += 1

        ho = self.hit_objects[self.ho_index]
        trajectory = torch.reshape(torch.tensor(get_trajectory(ho), dtype=torch.float32), (-1, 3))
        # Using first trajectory point for label time is very wrong for repeat sliders
        first_p = trajectory[0]
        label = get_coord_index3(ho.position)

        if self.prev_trajectory is not None:
            # Fade the previous img by the time diff
            img = torch.clip(self.prev_img - (first_p[2] - self.prev_trajectory[0, 2]) / self.look_back_time, 0, 1)

            # Draw the previous trajectory
            self.prev_img = draw_trajectory(img, self.prev_trajectory, first_p[2], self.look_back_time)

        self.prev_trajectory = trajectory
        self.ho_index += 1

        return self.prev_img.unsqueeze(0), label, self.circle_radius


class ImgBeatmapDatasetIterableFactory:
    __slots__ = "look_back_time"

    def __init__(self, look_back_time):
        self.look_back_time = look_back_time

    def __call__(self, *args, **kwargs):
        beatmap_files = args[0]
        return ImgBeatmapDatasetIterable(
            beatmap_files=beatmap_files,
            look_back_time=self.look_back_time,
        )


def get_img_data_loader(
        dataset_path: str,
        start: int,
        end: int,
        look_back_time: float = 5000,
        cycle_length=1,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        beatmap_files: Optional[list[str]] = None,
) -> DataLoader:
    return get_data_loader(
        dataset_path=dataset_path,
        start=start,
        end=end,
        iterable_factory=ImgBeatmapDatasetIterableFactory(
            look_back_time=look_back_time,
        ),
        cycle_length=cycle_length,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        drop_last=drop_last,
        beatmap_files=beatmap_files,
    )


def main(args):
    dataloader = get_img_data_loader(
        dataset_path=args.data_path,
        start=0,
        end=16291,
        look_back_time=5000,
        cycle_length=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
    )

    if args.mode == "plotfirst":
        import matplotlib.pyplot as plt

        i = 0
        for x, y in dataloader:
            print(x.shape, y.shape)
            print(x[0, 0])
            print(y)

            for j in range(args.batch_size):
                plot_signed_distance_field(x[j], y[j])
                plt.show()
                time.sleep(1)

            i += 1
            if i > 26:
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
