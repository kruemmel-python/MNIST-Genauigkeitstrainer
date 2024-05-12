# Importieren der notwendigen Bibliotheken für das Projekt
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Benutzerdefinierter Callback, um das Training bei einer Genauigkeit von 0.99 zu stoppen
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        accuracy = logs.get('accuracy')
        if accuracy is not None:
            if accuracy >= 0.99:
                print(f"\nGenauigkeit hat {accuracy:.4f} erreicht, breche das Training ab!")
                self.model.stop_training = True

my_callback = MyCallback()

# Laden des MNIST-Datensatzes, der in Trainings- und Testdaten aufgeteilt ist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Anpassen der Bildformate für das Training
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalisierung der Bildpixelwerte von 0-255 auf 0-1
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Umwandlung der Labels in One-Hot-Vektoren
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Erstellen eines ImageDataGenerator für Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(train_images)

# Festlegen des Pfades, wo das Modell gespeichert wird
model_path = 'mein_ki_modell.keras'

# Überprüfen, ob ein Modell bereits existiert, und dieses laden, falls vorhanden
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Modell geladen.")
else:
    # Erstellen eines neuen Modells, wenn keines existiert
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    print("Neues CNN-Modell erstellt.")

# Kompilieren des Modells
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callback für das Speichern des besten Modells
checkpoint = ModelCheckpoint(model_path, save_best_only=True)

# Trainieren des Modells mit den Trainingsdaten und Data Augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                    epochs=250,  # Setze eine hohe Epochenzahl
                    validation_data=(test_images, test_labels),
                    callbacks=[checkpoint, my_callback])  # Benutzerdefinierter Callback

# Evaluieren des Modells mit den Testdaten
test_loss, test_acc = model.evaluate(test_images, test_labels)
if test_acc >= 0.99:
    print(f"Testgenauigkeit: {test_acc:.4f}")
else:
    print("Testgenauigkeit hat 0.99 noch nicht erreicht.")

# Plotten der Trainings- und Validierungsgenauigkeit sowie der Verluste
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Trainingsgenauigkeit')
plt.plot(history.history['val_accuracy'], label='Validierungsgenauigkeit')
plt.title('Genauigkeit über Epochen')
plt.xlabel('Epochen')
plt.ylabel('Genauigkeit')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Trainingsverlust')
plt.plot(history.history['val_loss'], label='Validierungsverlust')
plt.title('Verlust über Epochen')
plt.xlabel('Epochen')
plt.ylabel('Verlust')
plt.legend()
plt.show()
