import torch
import torch.nn as nn
import numpy as np
from models import UNet3
from constants import image_shape
from data_loading_img import geometry_to_sdf, get_timestep_embedding, get_coord_index3
from plotting import plot_signed_distance_field, plot_prediction


def example_from_beatmap(beatmap):
    hit_objects = beatmap.hit_objects(spinners=False)
    prior = hit_objects[:-1]
    posterior = hit_objects[-1]
    label_time = posterior.time.total_seconds() * 1000
    label = get_coord_index3(posterior.position)

    # Draw the previous trajectory
    img = torch.zeros((playfield_height_num, playfield_width_num), dtype=torch.float32)
    for ho in prior:
        trajectory = torch.reshape(torch.tensor(get_trajectory(ho), dtype=torch.float32), (-1, 3))
        img = draw_trajectory(img, trajectory, label_time, 5000)

    return img.unsqueeze(0), label, get_hit_object_radius(beatmap.circle_size)


def load_example_folder(name):
    data = []
    for path in os.listdir(os.path.join("toy_datasets", name)):
        beatmap = Beatmap.from_path(path)
        example = example_from_beatmap(beatmap)
        data.append(example)
    return CachedDataset(data)


def get_dataloader(dataset):
    dataloader = DataLoader(
        dataset,
        pin_memory=True,
    )

    return dataloader


datasets = [
    "streams",
    "triangles",
]


def main(args):
    # Build model
    model = OsuModel.load_from_checkpoint(args.ckpt)

    wandb_logger = WandbLogger(
        project="sdf-osu",
        name=f"test {model.encoder_name} [{','.join(args.tests)}]",
        offline=args.offline,
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
    )

    for test in args.tests:
        test_dataloader = get_dataloader(load_example_folder(test))

        # TODO: add name of test to metrics
        trainer.test(
            model,
            dataloaders=test_dataloader,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--offline", type=bool, default=False)
    parser.add_argument("--tests", type=list[str], default=datasets.keys())
    args = parser.parse_args()
    main(args)
