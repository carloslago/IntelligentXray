import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from datetime import datetime
from functions import *
from tensorflow.keras.utils import plot_model

training_set = pd.read_csv("CheXpert-v1.0-small/csv/pathologies/train_all_3_3_mix.csv")
valid_set = pd.read_csv("CheXpert-v1.0-small/csv/pathologies/valid_mix.csv")

# types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
#          'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
#          'Fracture', 'Support_Devices']

types = ['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural_Effusion']

train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=True,
                                zoom_range=0.2,
                                # rotation_range=20,
                                # width_shift_range=.2,
                                # height_shift_range=.2,
                                # preprocessing_function=tf.keras.applications.densenet.preprocess_input
                                   )




# train_dataGen = ImageDataGenerator(rescale=1. / 255,
#                                 horizontal_flip=True,
#                                 zoom_range=0.2,
#                                rotation_range=25,
#                                width_shift_range=.2,
#                                height_shift_range=.2,
#                                 shear_range=.2,
#                                 brightness_range=[0.8,1.2]
#                                 # samplewise_center=True
#                                    )

datagen = ImageDataGenerator(rescale=1. / 255)
BATCH_SIZE = 24

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
model1 = tf.keras.models.Sequential([
    tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=inputShape),
    # tf.keras.layers.GlobalAveragePooling2D(activity_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.GlobalAveragePooling2D(),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(types), activation='sigmoid')
])
# plot_model(model1, to_file='model_complex.png', show_shapes=True)
# exit()

model1.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc', f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])
# model1.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc', tf.keras.metrics.AUC()])

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



