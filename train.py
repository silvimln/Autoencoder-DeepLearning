import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.callbacks import History
import tensorflow as tf
from model_autoencoder import build_grayscale_to_rgb_autoencoder  # Ganti nama kalau perlu

# Konstanta
ORIGINAL_DIR = 'dataset/original'
RESULTS_DIR = 'results'
IMG_SIZE = (128, 128)
EPOCHS = 100
BATCH_SIZE = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# LOAD DATA: Grayscale ke RGB
# ================================
def load_data_grayscale_to_rgb(original_dir, img_size):
    X_gray = []
    Y_rgb = []

    for fname in sorted(os.listdir(original_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(original_dir, fname)
        img = Image.open(img_path).convert('RGB').resize(img_size)

        rgb = np.array(img).astype(np.float32) / 255.0  # Normalisasi
        gray = np.mean(rgb, axis=2, keepdims=True)      # Konversi ke grayscale

        X_gray.append(gray)
        Y_rgb.append(rgb)

    return np.array(X_gray), np.array(Y_rgb)

# Load dataset
X, y = load_data_grayscale_to_rgb(ORIGINAL_DIR, IMG_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Build & Train Model
# ================================
model = build_grayscale_to_rgb_autoencoder(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))  # input grayscale
model.summary()

# Callback untuk menyimpan riwayat loss
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# ================================
# Visualisasi Plot Loss
# ================================
def plot_loss(history: History):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

# ================================
# Visualisasi Hasil Citra
# ================================
preds = model.predict(X_test[:5])
for i, (in_gray, out_rgb, true_rgb) in enumerate(zip(X_test[:5], preds, y_test[:5])):
    # Normalisasi input dan output untuk visualisasi
    input_img = (in_gray.squeeze() * 255).astype(np.uint8)
    output_img = (out_rgb * 255).astype(np.uint8)
    target_img = (true_rgb * 255).astype(np.uint8)

    # Simpan citra input, output, dan target
    Image.fromarray(input_img, mode='L').save(f'{RESULTS_DIR}/input_{i}.png')
    Image.fromarray(output_img).save(f'{RESULTS_DIR}/output_{i}.png')
    Image.fromarray(target_img).save(f'{RESULTS_DIR}/target_{i}.png')

    # Plot hasil visualisasi
    plt.figure(figsize=(12, 4))
    
    # Input Grayscale
    plt.subplot(1, 3, 1)
    plt.imshow(input_img, cmap='gray')
    plt.title('Input Grayscale')
    plt.axis('off')
    
    # Output RGB
    plt.subplot(1, 3, 2)
    plt.imshow(output_img)
    plt.title('Output RGB')
    plt.axis('off')

    # Target RGB
    plt.subplot(1, 3, 3)
    plt.imshow(target_img)
    plt.title('Target RGB')
    plt.axis('off')
    
    plt.show()
