# MNIST-Genauigkeitstrainer

Dieses Repository enth�lt ein TensorFlow-Projekt, das darauf abzielt, ein Convolutional Neural Network (CNN) zu trainieren, um handschriftliche Ziffern des MNIST-Datensatzes mit einer Genauigkeit von mindestens 99% zu erkennen.

## Projekt�bersicht

Das Projekt verwendet TensorFlow und Keras, um ein CNN-Modell zu erstellen und zu trainieren. Es beinhaltet Data Augmentation, um die Vielfalt der Trainingsdaten zu erh�hen und die Generalisierungsf�higkeit des Modells zu verbessern. Ein benutzerdefinierter Callback sorgt daf�r, dass das Training fortgesetzt wird, bis die gew�nschte Genauigkeit erreicht ist.

## Voraussetzungen

Bevor Sie beginnen, stellen Sie sicher, dass Sie folgende Software installiert haben:
- Python 3.6 oder h�her
- TensorFlow 2.x
- Matplotlib (f�r das Plotten der Trainingshistorie)

## Installation

Klonen Sie das Repository auf Ihren lokalen Computer:


Wechseln Sie in das Repository-Verzeichnis:


Installieren Sie die erforderlichen Python-Pakete:

## pip install -r requirements.txt


## Modelltraining

Um das Modell zu trainieren, f�hren Sie das Skript `train_model.py` aus:

python train_model.py




Das Skript wird das Modell trainieren und automatisch stoppen, sobald eine Genauigkeit von 99% erreicht ist.

## Ergebnisse

Nach dem Training k�nnen Sie die Testgenauigkeit �berpr�fen und die Trainingshistorie visualisieren, indem Sie die entsprechenden Abschnitte im Skript `train_model.py` ausf�hren.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die LICENSE Datei f�r Details.


## predict_and_display.py

Diese Datei enth�lt ein Skript, das eine grafische Benutzeroberfl�che (GUI) mit Tkinter erstellt. Es erm�glicht dem Benutzer, eine Zahlenfolge einzugeben, die dann vom trainierten Modell vorhergesagt wird. Die vorhergesagten Ziffern werden zusammen mit den entsprechenden MNIST-Bildern angezeigt.

Verwendung:
Starten Sie das Skript mit python predict_and_display.py.
Geben Sie eine Zahlenfolge ein, wenn Sie dazu aufgefordert werden.
Das Skript zeigt die Bilder und die vom Modell vorhergesagten Ziffern an.

## explore_model_data.py

Dieses Skript �ffnet die .keras Datei, die das trainierte Keras-Modell enth�lt, und druckt die Daten jedes Datensatzes innerhalb der Datei aus. Es ist n�tzlich, um die Struktur des Modells und die gespeicherten Gewichte zu verstehen.

Verwendung:
F�hren Sie das Skript mit python explore_model_data.py aus.
Das Skript gibt die Namen und Daten aller Datens�tze im Modell aus.



