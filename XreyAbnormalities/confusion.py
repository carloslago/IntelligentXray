import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.backend import manual_variable_initialization

np.random.seed(42)
tf.random.set_seed(42)
manual_variable_initialization(True)

path_model = os.path.join('saved_models/transfer_network.h5')
model = load_model(path_model, compile=False)
datagen = ImageDataGenerator(rescale=1 / 255)
test_dir = os.path.join('dataset/ChinaSet_AllFiles/test')
test_generator = datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)


filenames = test_generator.filenames
nb_samples = len(filenames)
Y_pred = model.predict_generator(test_generator, nb_samples/64)
y_pred = np.around(Y_pred)
y_test = test_generator.classes
# print(y_test)
# print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.xticks(range(2), ['Normal', 'Tuberculosis'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Tuberculosis'], fontsize=16)
plt.show()

