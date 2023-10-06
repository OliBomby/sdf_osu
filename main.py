import tensorflow as tf
import numpy as np
from tensorflow import keras
from models import get_model2
from constants import image_shape
from data_loading import process_path, to_images, list_beatmap_files_from_ds
from plotting import plot_signed_distance_field, plot_prediction

# Create training dataset
batch_size = 20
labeled_ds = list_beatmap_files_from_ds().interleave(process_path, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(500) \
    .map(to_images, num_parallel_calls=16)\
    .shuffle(1000)\
    .batch(batch_size, drop_remainder=True)\
    .prefetch(tf.data.AUTOTUNE)

# Build model
model = get_model2(image_shape, 1)
model.summary()

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy because our target data is integers.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

checkpoint_filepath = "saved models/sdf_osu_1_2.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=False)
]

model.load_weights(checkpoint_filepath)

# Train the model
epochs = 65
# model.fit(labeled_ds, epochs=epochs, steps_per_epoch=1000, callbacks=callbacks)


# Generate predictions for all images in the validation set
num_val = 10
val_ds = list(list_beatmap_files_from_ds().skip(10).interleave(process_path, cycle_length=1).map(to_images).skip(20).take(num_val))
val_preds = model.predict(tf.stack([img for (img, lab) in val_ds]))

for j in range(num_val):
    plot_signed_distance_field(val_ds[j][0].numpy(), val_ds[j][1])
    plot_prediction(np.sqrt(val_preds[j]))
