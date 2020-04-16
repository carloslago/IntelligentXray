import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras import models, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path, PureWindowsPath
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from keras.backend import manual_variable_initialization

np.random.seed(42)
tf.random.set_seed(42)
manual_variable_initialization(True)

# p = PureWindowsPath(Path().absolute())
# model = str(p.parents[0]) + '\\Pneuomonia\\saved_models\\best_network.h5'
path_model = os.path.join('saved_models/best_network.h5')


train_dir = os.path.join('dataset/ChinaSet_AllFiles/train')
test_dir = os.path.join('dataset/ChinaSet_AllFiles/test')

train_names = os.listdir(train_dir)

datagen = ImageDataGenerator(rescale=1 / 255)

# training_datagen = ImageDataGenerator(rescale=1. / 255,
#                                 horizontal_flip=True,
#                                 zoom_range=0.25,
#                                rotation_range=25,
#                                width_shift_range=.2,
#                                height_shift_range=.2,
#                                 shear_range=.2,
#                                 brightness_range=[0.8,1.2]
#                                 # samplewise_center=True
#                                    )
training_datagen = ImageDataGenerator(rescale=1/255, zoom_range=0.3, rotation_range=30)

train_generator = training_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)

test_generator = datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="grayscale",
    batch_size=64,
    class_mode="binary"
)

model = load_model(path_model, compile=False)

for layer in model.layers:
    if "core" not in str(layer): # Set convnet layers not trainable
        layer.trainable = False


# Changing dense layers
hidden = keras.layers.Dense(512, activation='sigmoid', name='my_dense3')(model.layers[-3].output)
# hidden2 = keras.layers.Dropout(0.5)(hidden)
# hidden2 = keras.layers.BatchNormalization(name="dasdad")(hidden)
new_layer = keras.layers.Dense(1, activation='sigmoid', name='my_dense')(hidden)
inp = model.input
# out = new_layer(model.layers[-2].output)
model2 = Model(inp, new_layer)

model2.summary()
# exit()
model2.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, decay=1e-5),
              metrics=['acc'])


early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, mode='min', verbose=1, restore_best_weights=True)

history = model2.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=8,
                    validation_steps=3,
                    epochs= 2,
                    verbose = 2,
                    callbacks=[early_stop]
)

model2.save('saved_models/transfer_network.h5')

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

filenames = test_generator.filenames
nb_samples = len(filenames)
Y_pred = model2.predict_generator(test_generator, nb_samples/64)
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