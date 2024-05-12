from tensorflow.keras.models import load_model

# Der relative Pfad zur .h5 Datei
file_path = 'mein_ki_modell - Kopie.keras'

# Das Keras-Modell laden
model = load_model(file_path)

# Die Architektur des Modells ausgeben
model.summary()

# Die Gewichte des Modells ausgeben
for layer in model.layers:
    weights = layer.get_weights()
    print(f"Gewichte für Schicht {layer.name}: {weights}")
