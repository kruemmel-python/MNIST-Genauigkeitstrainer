# MNIST-Genauigkeitstrainer

Dieses Repository enthält ein TensorFlow-Projekt, das darauf abzielt, ein Convolutional Neural Network (CNN) zu trainieren, um handschriftliche Ziffern des MNIST-Datensatzes mit einer Genauigkeit von mindestens 99% zu erkennen.

## Projektübersicht

Das Projekt verwendet TensorFlow und Keras, um ein CNN-Modell zu erstellen und zu trainieren. Es beinhaltet Data Augmentation, um die Vielfalt der Trainingsdaten zu erhöhen und die Generalisierungsfähigkeit des Modells zu verbessern. Ein benutzerdefinierter Callback sorgt dafür, dass das Training fortgesetzt wird, bis die gewünschte Genauigkeit erreicht ist.

## Voraussetzungen

Bevor Sie beginnen, stellen Sie sicher, dass Sie folgende Software installiert haben:
- Python 3.6 oder höher
- TensorFlow 2.x
- Matplotlib (für das Plotten der Trainingshistorie)

## Installation

Klonen Sie das Repository auf Ihren lokalen Computer:


Wechseln Sie in das Repository-Verzeichnis:


Installieren Sie die erforderlichen Python-Pakete:

## pip install -r requirements.txt


## Modelltraining

Um das Modell zu trainieren, führen Sie das Skript `train_model.py` aus:

python train_model.py




Das Skript wird das Modell trainieren und automatisch stoppen, sobald eine Genauigkeit von 99% erreicht ist.

## Ergebnisse

Nach dem Training können Sie die Testgenauigkeit überprüfen und die Trainingshistorie visualisieren, indem Sie die entsprechenden Abschnitte im Skript `train_model.py` ausführen.

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe die LICENSE Datei für Details.


## predict_and_display.py

Diese Datei enthält ein Skript, das eine grafische Benutzeroberfläche (GUI) mit Tkinter erstellt. Es ermöglicht dem Benutzer, eine Zahlenfolge einzugeben, die dann vom trainierten Modell vorhergesagt wird. Die vorhergesagten Ziffern werden zusammen mit den entsprechenden MNIST-Bildern angezeigt.

Verwendung:
Starten Sie das Skript mit python predict_and_display.py.
Geben Sie eine Zahlenfolge ein, wenn Sie dazu aufgefordert werden.
Das Skript zeigt die Bilder und die vom Modell vorhergesagten Ziffern an.

## explore_model_data.py

Dieses Skript öffnet die .keras Datei, die das trainierte Keras-Modell enthält, und druckt die Daten jedes Datensatzes innerhalb der Datei aus. Es ist nützlich, um die Struktur des Modells und die gespeicherten Gewichte zu verstehen.

Verwendung:
Führen Sie das Skript mit python explore_model_data.py aus.
Das Skript gibt die Namen und Daten aller Datensätze im Modell aus.



