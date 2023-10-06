import tensorflow as tf
import numpy as np
from models import get_model3
from constants import image_shape
from data_loading import geometry_to_sdf, get_timestep_embedding, get_coord_index3
from plotting import plot_signed_distance_field, plot_prediction

# Build model
model = get_model3(image_shape, 4)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

checkpoint_filepath = "saved models/sdf_osu_5_fine.h5"
model.load_weights(checkpoint_filepath)


def create_example(true, last1, last2, w1, w2, radius, time1, time2):
    positions = [last1, last2, w1, w2]
    return (np.stack(list(map(geometry_to_sdf, positions, (radius for _ in range(4)))), axis=2), get_timestep_embedding(time1), get_timestep_embedding(time2)), get_coord_index3(true)


def triangle_positions(x, y, size):
    return np.array([
        [x, y],
        [x, y + size],
        [x + size * np.sqrt(3) / 2, y + size / 2]
    ], dtype=np.float32)


def create_triangle_example(x, y, size, radius=36):
    true = triangle_positions(x, y, size)[-1]
    last1 = triangle_positions(x, y, size)[1]
    last2 = triangle_positions(x, y, size)[0]
    w1 = triangle_positions(x, y, size)[:2]
    w2 = triangle_positions(x, y + 100, size)
    return create_example(true, last1, last2, w1, w2, radius, 300, 300)


def stream_positions(x, y, size, spacing, count):
    t = np.linspace(0, spacing, count)
    return np.asarray(np.stack([np.cos(t), np.sin(t)], axis=-1) * size + np.array((x, y)), dtype=np.float32)


def create_stream_example(x, y, size, spacing, count, time, radius=36):
    pos = stream_positions(x, y, size, spacing, count)
    true = pos[-1]
    last1 = pos[-2]
    last2 = pos[-3]
    w1 = pos[(count - 1) // 2:(count - 1)]
    w2 = pos[:(count - 1) // 2]
    return create_example(true, last1, last2, w1, w2, radius, time, time)


def example_generator():
    # yield create_triangle_example(50, 50, 200)
    yield create_stream_example(256, 192, 180, np.pi, 15, 100)
    yield create_stream_example(256, 192, 180, 1.5 * np.pi, 15, 200)
    yield create_stream_example(256, 192, 180, 2 * np.pi, 15, 300)


def create_val_ds():
    ds = tf.data.Dataset.from_generator(
        example_generator,
        args=[],
        output_signature=(
            (
                tf.TensorSpec(image_shape + (4,), tf.float32),
                tf.TensorSpec(64, tf.float32),
                tf.TensorSpec(64, tf.float32),
            ),
            tf.TensorSpec((), tf.int32),
        )
    )

    return list(ds.batch(1000))[0]


# Generate predictions for all images in the validation set
val_ds = create_val_ds()
val_preds = model.predict(val_ds[0])

for j in range(len(val_ds[0][0])):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[0][0][j, :, :, i].numpy(), val_ds[1][j])
    plot_prediction(np.sqrt(val_preds[j]))
