import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
# from keras_preprocessing import image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from datetime import datetime
from functions import *
import numpy as np
from tensorflow.keras.utils import plot_model

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

training_set = pd.read_csv("CheXpert-v1.0-small/csv/pathologies/train_all_4_4_mix_ones.csv")
valid_set = pd.read_csv("CheXpert-v1.0-small/csv/pathologies/valid_mix.csv")

# types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
#          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
#          'Fracture', 'Support_Devices']

types = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']

train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=True,
                                zoom_range=0.2,
                                zca_whitening=True,
                                rotation_range=5,
                                   )


datagen = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 12
img_size = 224

dataframe_train = training_set
steps_train = len(dataframe_train) / BATCH_SIZE
steps_train = round(steps_train + 0.5)
train_generator = train_dataGen.flow_from_dataframe(
    dataframe=dataframe_train,
    directory="", x_col="Path",
    y_col=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE,
    shuffle=True)

test_set = valid_set
dataframe_valid = test_set
steps_valid = len(dataframe_valid) / BATCH_SIZE
steps_valid = round(steps_valid + 0.5)
valid_generator = datagen.flow_from_dataframe(
    dataframe=dataframe_valid,
    # dataframe = valid_set,
    directory="", x_col="Path",
    y_col=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(img_size, img_size),
    batch_size=BATCH_SIZE)

inputShape = (img_size, img_size, 3)
model1 = tf.keras.models.Sequential([
    tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=inputShape),
    tf.keras.layers.GlobalAveragePooling2D(activity_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(len(types), activation='sigmoid')
])
# plot_model(model1, to_file='model_complex.png', show_shapes=True)
# exit()

model1.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])

date = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")

csv_logger = CSVLogger('logs/log_all' + date + '.csv')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, mode='min', verbose=1, restore_best_weights=True)
model_path = 'saved_models/best_model_all' + date + '.h5'
mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1)
history = model1.fit_generator(train_generator, epochs=25,
                               steps_per_epoch=steps_train,
                               validation_data=valid_generator,
                               validation_steps=steps_valid,
                               verbose=2,
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



