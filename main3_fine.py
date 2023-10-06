import tensorflow as tf
import numpy as np
from tensorflow import keras
from models import get_model3
from constants import image_shape
from data_loading import list_beatmap_files_from_ds, process_path3, list_beatmap_files_from_ds_with_sr
from plotting import plot_signed_distance_field, plot_prediction

# Create training dataset
batch_size = 32
labeled_ds = list_beatmap_files_from_ds_with_sr(5, 15)\
    .skip(0) \
    .interleave(process_path3, cycle_length=16, num_parallel_calls=4) \
    .shuffle(1000) \
    .batch(batch_size, drop_remainder=True) \
    .prefetch(tf.data.AUTOTUNE)

# Build model
model = get_model3(image_shape, 4)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.summary()

checkpoint_filepath_old = "saved models/sdf_osu_5_fine.h5"
checkpoint_filepath = "saved models/sdf_osu_5_fine_2.h5"
callbacks = [
    keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor='loss')
]

model.load_weights(checkpoint_filepath_old)

# Train the model
epochs = 1000
model.fit(labeled_ds, epochs=epochs, steps_per_epoch=1000, callbacks=callbacks)


# Generate predictions for all images in the validation set
num_val = 5
val_ds = list(list_beatmap_files_from_ds_with_sr(5, 15).skip(19).interleave(process_path3, cycle_length=num_val).skip(100).batch(num_val).take(1))[0]
val_preds = model.predict(val_ds[0])

for j in range(num_val):
    for i in range(3, -1, -1):
        plot_signed_distance_field(val_ds[0][0][j, :, :, i].numpy(), val_ds[1][j])
    plot_prediction(np.sqrt(val_preds[j]))
