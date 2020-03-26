import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from datetime import datetime
from functions import *

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

train_dir = os.path.join('CheXpert-v1.0-small/train')
val_dir = os.path.join('CheXpert-v1.0-small/valid')
training_set = pd.read_csv("CheXpert-v1.0-small/csv/top/train_all_6_5.csv")
valid_set = pd.read_csv("CheXpert-v1.0-small/csv/original/valid_all.csv")

types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

'''Possible augmentations
    zca_whitening - Less redundancy in the image is intended to better highlight the structures and features in the image to the learning algorithm. related with PCA
    contrast_stretching - ontrast Stretching takes the approach of analyzing the distribution of pixel densities in an image and then “rescales the image to include all intensities that fall within the 2nd and 98th percentiles.”
    histogram_equalization - increases contrast in images by detecting the distribution of pixel densities in an image
    adaptive_equalization - differs from regular histogram equalization in that several different histograms are computed, each corresponding to a different section of the image;
     however, it has a tendency to over-amplify noise in otherwise uninteresting sections.

'''
train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=True,
                                zoom_range=0.2
                                   )

datagen = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 128

dataframe_train = training_set
steps_train = len(dataframe_train) / BATCH_SIZE
steps_train = round(steps_train + 0.5)
train_generator = train_dataGen.flow_from_dataframe(
    dataframe=dataframe_train,
    directory="", x_col="Path",
    y_col=types,
    class_mode="raw",
    color_mode="rgb",
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    shuffle=True)

# x,y = train_generator.next()
# for i in range(0,4):
#     image = x[i]
#     plt.imshow(image)
#     plt.show()

# test_set = pd.concat([training_set[5400:], valid_set])
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
    target_size=(224, 224),
    batch_size=BATCH_SIZE)

inputShape = (224, 224, 3)
# model1 = tf.keras.models.Sequential([
#     tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=inputShape),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(14, activation='sigmoid')
# ])

chanDim = -1

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=inputShape),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(3, 3),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(14, activation='sigmoid')
])

model2.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])

date = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")

csv_logger = CSVLogger('logs/log_all' + date + '.csv')
#min_delta = 0.1 - quiere decir que cada epoch debe mejorar un 0.1% por lo menos, vamos de 0.82 a 0.821
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=4, mode='min', verbose=1, restore_best_weights=True)
# early_stop = EarlyStopping(monitor='val_acc', baseline=0.85, patience=0, verbose=1)
model_path = 'saved_models/best_model_all' + date + '.h5'
mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1)
history = model2.fit_generator(train_generator, epochs=15,
                               steps_per_epoch=steps_train,
                               validation_data=valid_generator,
                               validation_steps=steps_valid,
                               callbacks=[csv_logger, mc])

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



