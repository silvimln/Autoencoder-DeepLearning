from keras import layers, models

def build_grayscale_to_rgb_autoencoder(input_shape=(128, 128, 1)):
    inp = layers.Input(shape=input_shape)

    # ========== Encoder (3 Blok) ==========
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 64x64

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 32x32

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)  # 16x16

    # ========== Decoder (3 Blok) ==========
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)  # 32x32

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 64x64

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)  # 128x128

    # Output layer — RGB (3 channel)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = models.Model(inputs=inp, outputs=decoded, name='GrayscaleToRGB_Autoencoder_3Block')
    model.compile(optimizer='adam', loss='mse')

    return model
