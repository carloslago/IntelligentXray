import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from datetime import datetime
from CheXpert.functions import *
from tensorflow.keras.utils import plot_model

training_set = pd.read_csv("dataset/train.csv")
valid_set = pd.read_csv("dataset/test.csv")

types = ['covid', 'no-finding', 'pneuomonia']

train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=True,
                                zoom_range=0.25,
                               rotation_range=25,
                               width_shift_range=.2,
                               height_shift_range=.2,
                                shear_range=.2,
                                brightness_range=[0.8,1.2]
                                # samplewise_center=True
                                   )

datagen = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 16

dataframe_train = training_set
steps_train = len(dataframe_train) / BATCH_SIZE
steps_train = round(steps_train + 0.5)
train_generator = train_dataGen.flow_from_dataframe(
    dataframe=dataframe_train,
    directory="", x_col="path",
    y_col=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True)


dataframe_valid = valid_set
steps_valid = len(dataframe_valid) / BATCH_SIZE
steps_valid = round(steps_valid + 0.5)
valid_generator = datagen.flow_from_dataframe(
    dataframe=dataframe_valid,
    directory="", x_col="path",
    y_col=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(224, 224),
    batch_size=BATCH_SIZE)

inputShape = (224, 224, 3)
model1 = tf.keras.models.Sequential([
    tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=inputShape),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

# plot_model(model1, to_file='model_dropout.png', show_shapes=True)
# exit()

model1.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])

date = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")

csv_logger = CSVLogger('logs/log_' + date + '.csv')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, mode='min', restore_best_weights=True)
model_path = 'saved_models/best_model_' + date + '.h5'
mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1)
history = model1.fit_generator(train_generator, epochs=20,
                               steps_per_epoch=steps_train,
                               validation_data=valid_generator,
                               validation_steps=steps_valid,
                               callbacks=[csv_logger, mc, early_stop])

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



