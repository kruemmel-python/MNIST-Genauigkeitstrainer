import tkinter as tk
from tkinter import simpledialog
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Laden des MNIST-Datensatzes
(_, _), (test_images, _) = mnist.load_data()

# Normalisierung der Bilder auf Werte zwischen 0 und 1
test_images = test_images.astype('float32') / 255

# Pfad für das geladene Modell festlegen
model_path = 'mein_ki_modell.keras'
model = tf.keras.models.load_model(model_path)

# Funktion zur Vorhersage und Anzeige der Bilder
def vorhersage_und_anzeige(zahlenfolge, test_images):
    # Überprüfen, ob mehr als ein Bild angezeigt werden soll
    if len(zahlenfolge) > 1:
        fig, axs = plt.subplots(1, len(zahlenfolge), figsize=(10, 2))
        for i, zahl in enumerate(zahlenfolge):
            # Umwandlung der Zahl in ein Bild
            bild = test_images[zahl].reshape(1, 28, 28, 1)
            # Vorhersage des Modells
            vorhersage = model.predict(bild)
            # Umwandlung der Vorhersage in eine lesbare Ziffer
            ziffer = np.argmax(vorhersage, axis=1)
            # Anzeige des Bildes und der vorhergesagten Ziffer
            axs[i].imshow(test_images[zahl], cmap='gray')
            axs[i].set_title(f'{ziffer[0]}')
            axs[i].axis('off')
    else:
        # Wenn nur ein Bild angezeigt werden soll
        fig, ax = plt.subplots(figsize=(2, 2))
        zahl = zahlenfolge[0]
        bild = test_images[zahl].reshape(1, 28, 28, 1)
        vorhersage = model.predict(bild)
        ziffer = np.argmax(vorhersage, axis=1)
        ax.imshow(test_images[zahl], cmap='gray')
        ax.set_title(f'{ziffer[0]}')
        ax.axis('off')
    plt.show()


# Erstellen der GUI
root = tk.Tk()
root.geometry("300x100")

def get_input():
    # Dialog zum Eingeben der Zahl
    answer = simpledialog.askstring("Eingabe", "Bitte geben Sie eine Zahl ein:", parent=root)
    if answer is not None:
        # Umwandlung der Eingabe in eine Liste von Indizes
        zahlenfolge = [int(digit) for digit in str(answer)]
        # Aufruf der Vorhersage- und Anzeigefunktion
        vorhersage_und_anzeige(zahlenfolge, test_images)

# Button zum Öffnen des Dialogs
button = tk.Button(root, text="Zahl eingeben", command=get_input)
button.pack(pady=20)

# Starten der GUI
root.mainloop()
