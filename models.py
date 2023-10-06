from abc import ABC

from tensorflow import keras

layers = keras.layers


def get_model1(img_size):
    inputs = keras.Input(shape=img_size + (1,))

    # [First half of the network: down-sampling inputs] ###

    # Entry block
    c = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    c = layers.BatchNormalization()(c)

    previous_block_activation = c  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        c = layers.Activation("relu")(c)
        c = layers.SeparableConv2D(filters, 3, padding="same")(c)
        c = layers.BatchNormalization()(c)

        c = layers.Activation("relu")(c)
        c = layers.SeparableConv2D(filters, 3, padding="same")(c)
        c = layers.BatchNormalization()(c)

        c = layers.MaxPooling2D(3, strides=2, padding="same")(c)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        c = layers.add([c, residual])  # Add back residual
        previous_block_activation = c  # Set aside next residual

    # [Second half of the network: up-sampling inputs] ###

    for filters in [256, 128, 64, 32]:
        c = layers.Activation("relu")(c)
        c = layers.Conv2DTranspose(filters, 3, padding="same")(c)
        c = layers.BatchNormalization()(c)

        c = layers.Activation("relu")(c)
        c = layers.Conv2DTranspose(filters, 3, padding="same")(c)
        c = layers.BatchNormalization()(c)

        c = layers.UpSampling2D(2)(c)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        c = layers.add([c, residual])  # Add back residual
        previous_block_activation = c  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(1, 3, padding="same")(c)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Softmax()(outputs)

    # Define the model
    return keras.Model(inputs, outputs, name="U-Net1")


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(n_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def get_model2(img_size, channels):
    # inputs
    inputs = keras.Input(shape=img_size + (channels,))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(1, 1, padding="same")(u9)
    outputs = layers.Flatten()(outputs)
    outputs = layers.Softmax()(outputs)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net2")
    return unet_model


def downsample_block3(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    # p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block3(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    # x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def get_model3(img_size, channels):
    inputs = keras.Input(shape=img_size + (channels,))
    t1_embed_input = keras.layers.Input((64,))
    t2_embed_input = keras.layers.Input((64,))

    t1_emb = keras.layers.Dense(128)(t1_embed_input)
    t1_emb = keras.layers.Activation("swish")(t1_emb)
    t2_emb = keras.layers.Dense(128)(t2_embed_input)
    t2_emb = keras.layers.Activation("swish")(t2_emb)
    t_emb = keras.layers.concatenate([t1_emb, t2_emb])
    t_emb = keras.layers.Dense(512)(t_emb)

    outputs = []
    x = inputs

    # encoder: contracting path - downsample
    for filters in [32, 64, 128, 256]:
        f, x = downsample_block3(x, filters)
        outputs.append(f)

    x = double_conv_block(x, 512)
    x = x + t_emb[:, None, None]

    # decoder: expanding path - upsample
    for filters in [256, 128, 64, 32]:
        x = upsample_block3(x, outputs.pop(), filters)

    # outputs
    output = layers.Conv2D(1, 3, padding="same")(x)
    output = layers.Flatten()(output)
    output = layers.Softmax()(output)

    # unet model with Keras Functional API
    return keras.Model([inputs, t1_embed_input, t2_embed_input], output, name="U-Net3")
