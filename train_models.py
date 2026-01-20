import tensorflow as keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import os
import time

if not os.path.exists('modele'):
    os.makedirs('modele')

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images_cnn = train_images.reshape((-1, 28, 28, 1))
test_images_cnn = test_images.reshape((-1, 28, 28, 1))

model_simple = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_simple.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

start_time_simple = time.time()
history_simple = model_simple.fit(train_images, train_labels, epochs=5, validation_split=0.1)
end_time_simple = time.time()
model_simple.save('modele/simple_model.h5')

model_advanced = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_advanced.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

start_time_adv = time.time()
history_advanced = model_advanced.fit(train_images_cnn, train_labels, epochs=10, validation_split=0.1)
end_time_adv = time.time()
model_advanced.save('modele/advanced_model.h5')

print("\n" + "="*40)
print("       DANE DO RAPORTU")
print("="*40)

acc_s = history_simple.history['val_accuracy'][-1] * 100
time_s = end_time_simple - start_time_simple
print(f"1. MODEL PROSTY (MLP):")
print(f"   - Dokładność: {acc_s:.2f}%")
print(f"   - Czas uczenia: {time_s:.1f} sekund")

acc_a = history_advanced.history['val_accuracy'][-1] * 100
time_a = end_time_adv - start_time_adv
print(f"\n2. MODEL ZAAWANSOWANY (CNN):")
print(f"   - Dokładność: {acc_a:.2f}%")
print(f"   - Czas uczenia: {time_a:.1f} sekund")
print("="*40)