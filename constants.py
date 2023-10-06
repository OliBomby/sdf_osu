import pathlib
import tensorflow as tf

data_root = pathlib.Path("/media/data/Osu! Dingen/Beatmap ML Datasets/ORS10548")
max_sdf_distance = 20
playfield_width = 512
playfield_height = 384
playfield_width_num = playfield_width // 4
playfield_height_num = playfield_height // 4
flat_num = playfield_height_num * playfield_width_num
image_shape = (playfield_height_num, playfield_width_num)

x = tf.linspace(0.0, playfield_width, playfield_width_num)
y = tf.linspace(0.0, playfield_height, playfield_height_num)
X, Y = tf.meshgrid(x, y)
coordinates = tf.reshape(tf.concat(
    (
        X[..., tf.newaxis],
        Y[..., tf.newaxis],
    ),
    axis=-1,
), (playfield_height_num, playfield_width_num, 1, 2))
coordinates_flat = tf.reshape(coordinates, (flat_num, 2))