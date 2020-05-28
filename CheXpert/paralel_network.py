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


types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

def generate_generator_multiple(generator, dt1, dt2, batch_size, img_height, img_width):
    genX1 = generator.flow_from_dataframe(dataframe=dt1,
                                          directory="",
                                          x_col="Path",
                                          y_col=types,
                                          class_mode="raw",
                                          color_mode="rgb",
                                          target_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          shuffle=False)

    genX2 = generator.flow_from_dataframe(dataframe=dt2,
                                          directory="",
                                          x_col="Path",
                                          y_col=types,
                                          class_mode="raw",
                                          color_mode="rgb",
                                          target_size=(img_height, img_width),
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


train_dataGen = ImageDataGenerator(rescale=1. / 255,
                                horizontal_flip=True,
                                zoom_range=0.25,
                               rotation_range=15,
                               width_shift_range=.2,
                               height_shift_range=.2,
                                shear_range=.2,
                                brightness_range=[0.8,1.2]
                                   )

test_imgen = ImageDataGenerator(rescale = 1./255)

training_set_frontal = pd.read_csv("CheXpert-v1.0-small/csv/paralel/train_paralel_frontal.csv")
training_set_lateral = pd.read_csv("CheXpert-v1.0-small/csv/paralel/train_paralel_lateral.csv")
valid_set_frontal = pd.read_csv("CheXpert-v1.0-small/csv/paralel/past/valid_paralel_frontal.csv")
valid_set_lateral = pd.read_csv("CheXpert-v1.0-small/csv/paralel/past/valid_paralel_lateral.csv")



BATCH_SIZE = 32

dataframe_train = training_set_frontal
steps_train = len(dataframe_train) / BATCH_SIZE
steps_train = round(steps_train + 0.5)


test_set = valid_set_frontal
dataframe_valid = test_set
steps_valid = len(dataframe_valid) / BATCH_SIZE
steps_valid = round(steps_valid + 0.5)

inputShape = (224, 224, 3)
chanDim = -1
model1 = tf.keras.models.Sequential([
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
])

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
])

combined = tf.keras.layers.concatenate([model1.output, model2.output])

# z = tf.keras.layers.BatchNormalization()
z = tf.keras.layers.Dense(512, activation='relu') (combined)
z = tf.keras.layers.BatchNormalization() (z)
z = tf.keras.layers.Dense(14, activation='sigmoid') (z)

model = tf.keras.Model(inputs=[model1.input, model2.input], outputs=z)

# plot_model(model, to_file='model_parallel.png', show_shapes=True)
# exit()
# print(model.summary())

inputgenerator=generate_generator_multiple(generator=train_dataGen,
                                           dt1=training_set_frontal,
                                           dt2=training_set_lateral,
                                           batch_size=BATCH_SIZE,
                                           img_height=224,
                                           img_width=224)

testgenerator=generate_generator_multiple(generator=test_imgen,
                                           dt1=valid_set_frontal,
                                           dt2=valid_set_lateral,
                                           batch_size=BATCH_SIZE,
                                           img_height=224,
                                           img_width=224)



model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['acc',  f1_m, precision_m, recall_m, tf.keras.metrics.AUC()])

date = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")

csv_logger = CSVLogger('logs/log_paralel' + date + '.csv')
#min_delta = 0.1 - quiere decir que cada epoch debe mejorar un 0.1% por lo menos, vamos de 0.82 a 0.821
# early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=3, mode='min', verbose=1, restore_best_weights=True)
# early_stop = EarlyStopping(monitor='val_acc', baseline=0.85, patience=0, verbose=1)
model_path = 'saved_models/best_model_paralel' + date + '.h5'
mc = ModelCheckpoint(model_path, monitor='val_loss', mode='min', verbose=1)
history = model.fit_generator(inputgenerator, epochs=25,
                               steps_per_epoch=steps_train,
                               validation_data=testgenerator,
                               validation_steps=steps_valid,
                                verbose=2,
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



