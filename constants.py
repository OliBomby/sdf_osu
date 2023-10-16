import torch
import pathlib

max_sdf_distance = 20
playfield_width = 512
playfield_height = 384
playfield_width_num = playfield_width // 4
playfield_height_num = playfield_height // 4
flat_num = playfield_height_num * playfield_width_num
image_shape = (playfield_height_num, playfield_width_num)

x = torch.linspace(0.0, playfield_width, playfield_width_num)
y = torch.linspace(0.0, playfield_height, playfield_height_num)
X, Y = torch.meshgrid(x, y, indexing="ij")
coordinates = torch.cat(
    (
        X.unsqueeze(-1),
        Y.unsqueeze(-1),
    ),
    dim=-1,
).reshape(playfield_height_num, playfield_width_num, 1, 2)
coordinates_flat = coordinates.reshape(flat_num, 2)
