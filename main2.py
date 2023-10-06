import tensorflow as tf
import numpy as np
from tensorflow import keras
from models import get_model2
from constants import image_shape
from data_loading import process_path2, to_images2, list_beatmap_files_from_ds
from plotting import plot_signed_distance_field, plot_prediction

# Create training dataset
batch_size = 20
labeled_ds = list_beatmap_files_from_ds()\
    .interleave(process_path2, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE) \
    .prefetch(500) \
    .map(to_images2, num_parallel_calls=16) \
    .shuffle(1000) \
    .batch(batch_size, drop_remainder=True) \
    .prefetch(tf.data.AUTOTUNE)

# Build model
model = get_model2(image_shape, 4)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

checkpoint_filepath = "saved models/sdf_osu_2.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=False, save_weights_only=True)
]

model.load_weights(checkpoint_filepath)

# Train the model
epochs = 70
model.fit(labeled_ds, epochs=epochs, steps_per_epoch=1000, callbacks=callbacks)


# Generate predictions for all images in the validation set
num_val = 5
val_ds = list(list_beatmap_files_from_ds().skip(19).interleave(process_path2, cycle_length=num_val).map(to_images2).skip(100).take(num_val))
val_preds = model.predict(tf.stack([img for (img, lab) in val_ds]))

for j in range(num_val):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[j][0][:, :, i].numpy(), val_ds[j][1])
    plot_prediction(np.sqrt(val_preds[j]))
