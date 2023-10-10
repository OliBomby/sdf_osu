import pathlib
import glob
import json
import math
import numpy as np
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from slider.beatmap import Beatmap, HitObject, Slider
from constants import coordinates_flat, coordinates, playfield_width_num, playfield_height_num, \
    max_sdf_distance, image_shape

empty_pos_tensor = torch.zeros((0, 2), dtype=torch.float32)
empty_sdf_array = torch.full(image_shape, max_sdf_distance, dtype=torch.float32)


def get_data(ho: HitObject):
    if isinstance(ho, Slider):
        return ho.end_time, torch.reshape(
            torch.tensor([ho.curve(t) for t in np.linspace(0, 1, 50)], dtype=torch.float32), (-1, 2))

    return ho.time, torch.tensor(ho.position, dtype=torch.float32).reshape(1, 2)


def get_hit_object_radius(circle_size):
    return (109 - 9 * circle_size) / 2


def read_and_process_beatmap(file_path):
    look_back_time = 2000
    beatmap: Beatmap = Beatmap.from_path(file_path)
    radius = get_hit_object_radius(beatmap.circle_size)
    hit_objects = beatmap.hit_objects(spinners=False)
    hit_object_data = [get_data(ho) for ho in hit_objects]

    # Make training examples with current object, 2s of previous objects, and next object as label
    buffer = []
    for i in range(0, len(hit_object_data) - 1):
        current_data = hit_object_data[i]
        next_data = hit_object_data[i + 1]

        while len(buffer) > 0 and buffer[0][0] < current_data[0] - timedelta(milliseconds=look_back_time):
            buffer.pop(0)

        buffer.append(current_data)

        yield torch.cat([data[1] for data in buffer], dim=0), \
            radius, \
            next_data[1]


def process_path(file_path):
    ds_positions = torch.utils.data.Dataset.from_generator(
        read_and_process_beatmap,
        args=[file_path],
        output_signature=(torch.TensorSpec((None, 2), torch.float32),
                          torch.TensorSpec(()),
                          torch.TensorSpec((None, 2), torch.float32),
                          )
    )
    return ds_positions


def batch_to_images(geometries, radii, next_positions):
    return torch.vectorized_map(to_images_vectorizable, (geometries, radii, next_positions), False)


def to_images_vectorizable(arg):
    geometry, radius, next_positions = arg
    return to_images(geometry, radius, next_positions)


def to_images(geometry, radius, next_positions):
    sdf = geometry_to_sdf(geometry, radius)
    label_index = get_coord_index(next_positions[0])
    return torch.unsqueeze(sdf, 2), label_index


def get_coord_index(pos):
    return torch.argmin(torch.sum(torch.square(coordinates_flat - pos), dim=-1))


def geometry_to_sdf(geometry, radius):
    """
    Calculates the playfield SDF from a list of positions representing the geometry, normalized to the radius
    :param geometry: Tensor of shape (M, 2) representing the positions
    :param radius: Scalar representing the radius of the geometry
    :return: The SDF
    """
    return torch.minimum(
        torch.sqrt(
            torch.reduce_min(
                torch.reduce_sum(
                    torch.square(
                        torch.tile(
                            torch.reshape(
                                geometry,
                                (1, 1, -1, 2)
                            ),
                            (playfield_height_num, playfield_width_num, 1, 1)
                        ) - coordinates),
                    dim=-1
                ),
                dim=-1
            )
        ) / radius - 1,
        max_sdf_distance
    )


def read_and_process_beatmap2(file_path):
    look_back_time = 2000
    beatmap: Beatmap = Beatmap.from_path(file_path)
    radius = get_hit_object_radius(beatmap.circle_size)
    hit_objects = beatmap.hit_objects(spinners=False)
    hit_object_data = [get_data(ho) for ho in hit_objects]

    # Make training examples with current object, 2s of previous objects, and next object as label
    num_buffers = 4
    buffer_width = timedelta(milliseconds=look_back_time / num_buffers)
    buffers = [[] for _ in range(num_buffers)]
    for i in range(0, len(hit_object_data) - 1):
        current_data = hit_object_data[i]
        next_data = hit_object_data[i + 1]

        for j in range(num_buffers - 1, -1, -1):
            buffer = buffers[j]
            while len(buffer) > 0 and buffer[0][0] < current_data[0] - (j + 1) * buffer_width:
                popped = buffer.pop(0)
                new_index = int((current_data[0] - popped[0]) / buffer_width)
                if new_index < num_buffers:
                    buffers[new_index].append(popped)

        buffers[0].append(current_data)

        yield buffer_to_point_tensor(buffers[0]), \
            buffer_to_point_tensor(buffers[1]), \
            buffer_to_point_tensor(buffers[2]), \
            buffer_to_point_tensor(buffers[3]), \
            radius, \
            next_data[1]


def buffer_to_point_tensor(buffer):
    return torch.cat([data[1] for data in buffer], dim=0) if len(buffer) > 0 else empty_pos_tensor


def process_path2(file_path):
    ds_positions = torch.utils.data.Dataset.from_generator(
        read_and_process_beatmap2,
        args=[file_path],
        output_signature=(torch.TensorSpec((None, 2), torch.float32),
                          torch.TensorSpec((None, 2), torch.float32),
                          torch.TensorSpec((None, 2), torch.float32),
                          torch.TensorSpec((None, 2), torch.float32),
                          torch.TensorSpec(()),
                          torch.TensorSpec((None, 2), torch.float32),
                          )
    )
    return ds_positions


def to_images2(g1, g2, g3, g4, radius, next_positions):
    sdf = torch.stack((
        geometry_to_sdf(g1, radius),
        geometry_to_sdf(g2, radius),
        geometry_to_sdf(g3, radius),
        geometry_to_sdf(g4, radius),
    ), dim=2)
    label_index = get_coord_index(next_positions[0])
    return sdf, label_index


def read_and_process_beatmap3(file_path):
    beatmap: Beatmap = Beatmap.from_path(file_path)
    radius = get_hit_object_radius(beatmap.circle_size)
    hit_objects = beatmap.hit_objects(spinners=False)

    # Make training examples with current object, 2s of previous objects, and next object as label
    num_buffers = 2
    look_back_time = 1500
    buffer_width = timedelta(milliseconds=look_back_time / num_buffers)
    buffers = [[] for _ in range(num_buffers)]
    last_time = None
    last_pos = None
    last_last_pos_sdf = empty_sdf_array
    last_time_since_last_emb = get_timestep_embedding(1000)
    for ho in hit_objects:
        start_time = ho.time
        end_time, data = get_data3(ho)
        sdf = geometry_to_sdf3(data, radius)
        label = get_coord_index3(data[0])

        for j in range(num_buffers - 1, -1, -1):
            buffer = buffers[j]
            while len(buffer) > 0 and buffer[0][0] < start_time - (j + 1) * buffer_width:
                popped = buffer.pop(0)
                new_index = int((start_time - popped[0]) / buffer_width)
                if new_index < num_buffers:
                    buffers[new_index].append(popped)

        last_pos_sdf = geometry_to_sdf3(last_pos, radius) if last_pos is not None else empty_sdf_array

        time_since_last = (start_time - last_time).total_seconds() * 1000 if last_time is not None else 1000
        time_since_last = np.clip(time_since_last, 0, 1000)
        time_since_last_emb = get_timestep_embedding(time_since_last)

        if time_since_last <= 93.75:  # 160 BPM stream
            yield (
            buffers_to_sdf_tensor([[(0, last_pos_sdf)], [(0, last_last_pos_sdf)]] + buffers), time_since_last_emb,
            last_time_since_last_emb), label

        buffers[0].append((end_time, sdf))
        last_time_since_last_emb = time_since_last_emb
        last_last_pos_sdf = last_pos_sdf
        last_time = end_time
        last_pos = data[-1]


def get_coord_index3(pos):
    return np.argmin(np.sum(np.square(coordinates_flat - pos), axis=-1))


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


def get_data3(ho: HitObject):
    if isinstance(ho, Slider):
        return ho.end_time, np.array([ho.curve(t) for t in np.linspace(0, 1, 50)], dtype=np.float32)

    return ho.time, np.array([ho.position], dtype=np.float32)


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


def process_path3(file_path):
    ds_positions = torch.utils.data.Dataset.from_generator(
        read_and_process_beatmap3,
        args=[file_path],
        output_signature=(
            (
                torch.TensorSpec(image_shape + (4,), torch.float32),
                torch.TensorSpec(64, torch.float32),
                torch.TensorSpec(64, torch.float32),
            ),
            torch.TensorSpec((), torch.int32),
        )
    )
    return ds_positions


def get_sdf_data_loader(
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
        subset_ids: list[int] | None = None,
) -> DataLoader:
    dataset = BeatmapDataset(
        dataset_path=dataset_path,
        start=start,
        end=end,
        iterable_fn=lambda beatmap_files: BeatmapDatasetIterable(
            beatmap_files=beatmap_files,
            seq_len=seq_len,
            stride=stride,
            seq_func=seq_func,
            win_func=win_func,
        ),
        cycle_length=cycle_length,
        shuffle=shuffle,
        subset_ids=subset_ids,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        worker_init_fn=worker_init_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return dataloader


def main(args):
    ds = list_beatmap_files_from_ds_with_sr(5, 15) \
        .interleave(process_path3, cycle_length=16, num_parallel_calls=16)

    # from plotting import plot_signed_distance_field
    # for f in ds.skip(200).take(1):
    #     print(f[0][0].shape, f[0][1], f[0][2], f[1])
    #     for g in range(f[0][0].shape[2]):
    #         plot_signed_distance_field(f[0][0][:, :, g].numpy())

    import time

    count = 0
    start = None
    for f in ds:
        if start is None:
            start = time.time()
        count += 1
        print(f"\r{count}, {count / (time.time() - start)} per second", end='')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
