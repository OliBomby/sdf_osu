import torch
import torch.nn as nn
import numpy as np
from models import UNet3
from constants import image_shape
from data_loading import geometry_to_sdf, get_timestep_embedding, get_coord_index3
from plotting import plot_signed_distance_field, plot_prediction

# Define your UNet3 model here (if not already defined)
# from models import get_model3

# Build model
model = UNet3(image_shape, 4)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

checkpoint_filepath = "saved models/sdf_osu_5_fine.pth"
checkpoint = torch.load(checkpoint_filepath)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']


def create_example(true, last1, last2, w1, w2, radius, time1, time2):
    positions = [last1, last2, w1, w2]
    sdf_maps = torch.stack([geometry_to_sdf(positions[k], radius) for k in range(4)], dim=2)
    return (sdf_maps, get_timestep_embedding(time1), get_timestep_embedding(time2)), get_coord_index3(true)


def triangle_positions(x, y, size):
    return torch.tensor([
        [x, y],
        [x, y + size],
        [x + size * np.sqrt(3) / 2, y + size / 2]
    ], dtype=torch.float32)


def create_triangle_example(x, y, size, radius=36):
    true = triangle_positions(x, y, size)[-1]
    last1 = triangle_positions(x, y, size)[1]
    last2 = triangle_positions(x, y, size)[0]
    w1 = triangle_positions(x, y, size)[:2]
    w2 = triangle_positions(x, y + 100, size)
    return create_example(true, last1, last2, w1, w2, radius, 300, 300)


def stream_positions(x, y, size, spacing, count):
    t = np.linspace(0, spacing, count)
    return torch.tensor(
        np.asarray(np.stack([np.cos(t), np.sin(t)], axis=-1) * size + np.array((x, y)), dtype=np.float32))


def create_stream_example(x, y, size, spacing, count, time, radius=36):
    pos = stream_positions(x, y, size, spacing, count)
    true = pos[-1]
    last1 = pos[-2]
    last2 = pos[-3]
    w1 = pos[(count - 1) // 2:(count - 1)]
    w2 = pos[:(count - 1) // 2]
    return create_example(true, last1, last2, w1, w2, radius, time, time)


def example_generator():
    yield create_stream_example(256, 192, 180, np.pi, 15, 100)
    yield create_stream_example(256, 192, 180, 1.5 * np.pi, 15, 200)
    yield create_stream_example(256, 192, 180, 2 * np.pi, 15, 300)


def create_val_ds():
    examples = list(example_generator())
    sdf_maps = torch.stack([e[0][0] for e in examples], dim=0)
    time1 = torch.stack([e[0][1] for e in examples], dim=0)
    time2 = torch.stack([e[0][2] for e in examples], dim=0)
    labels = torch.tensor([e[1] for e in examples], dtype=torch.int32)

    return (sdf_maps, time1, time2), labels


# Generate predictions for all images in the validation set
val_ds = create_val_ds()
model.eval()  # Set model to evaluation mode
val_preds = []

with torch.no_grad():
    inputs = (val_ds[0][0], val_ds[0][1], val_ds[0][2])
    outputs = model(inputs)
    val_preds.append(outputs.numpy())

for j in range(len(val_ds[0])):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[0][j][:, :, i].numpy(), val_ds[1][j])
    plot_prediction(np.sqrt(val_preds[0][j]))
