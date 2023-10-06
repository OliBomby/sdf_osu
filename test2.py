import tensorflow as tf
import numpy as np
from tensorflow import keras
from models import get_model2
from constants import image_shape, coordinates_flat
from data_loading import process_path2, to_images2, list_beatmap_files_from_ds, geometry_to_sdf
from plotting import plot_signed_distance_field, plot_prediction

# Build model
model = get_model2(image_shape, 4)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

checkpoint_filepath = "saved models/sdf_osu_2.h5"
model.load_weights(checkpoint_filepath)


def triangle_positions(x, y, size):
    return tf.constant([
        [x, y],
        [x, y + size],
        [x + size * np.sqrt(3) / 2, y + size / 2]
    ], dtype=tf.float32)


def create_triangle_example(x, y, size, num_examples, radius=36):
    last, true = triangle_positions(512 - x, y + size, -size)[:2], triangle_positions(x, y, size)[-1]
    positions = [last] + [triangle_positions(x, y, size) if k < num_examples else tf.constant(0, shape=(0, 2), dtype=tf.float32) for k in range(3)]
    return np.stack(list(map(geometry_to_sdf, positions, (radius for _ in range(4)))), axis=2), np.argmin(np.linalg.norm(coordinates_flat - true))


def create_val_ds():
    examples = [
        create_triangle_example(50, 50, 100, 3),
        create_triangle_example(50, 50, 200, 3),
        create_triangle_example(50, 50, 300, 3),
    ]

    return examples


# Generate predictions for all images in the validation set
val_ds = create_val_ds()
val_preds = model.predict(tf.stack([img for (img, lab) in val_ds]))

for j in range(len(val_ds)):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[j][0][:, :, i], val_ds[j][1])
    plot_prediction(np.sqrt(val_preds[j]))
