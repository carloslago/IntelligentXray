import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop, Adam
import pandas as pd
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from datetime import datetime

# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

train_dir = os.path.join('dataset/train')
val_dir = os.path.join('dataset/valid')
training_set = pd.read_csv("dataset/csv/top/train_top_lateral.csv")
valid_set = pd.read_csv("dataset/csv/original/valid_lateral.csv")

types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural_Other',
         'Fracture', 'Support_Devices']

train_dataGen = ImageDataGenerator(rescale = 1./255, horizontal_flip=True, featurewise_std_normalization=True)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_dataGen.flow_from_dataframe(
                                        dataframe = training_set,
                                        directory="",x_col="Path",
                                        y_col=types,
                                        class_mode="raw",
                                        color_mode="rgb",
                                        target_size=(224,224),
                                        batch_size=8)

# x,y = train_generator.next()
# for i in range(0,4):
#     image = x[i]
#     plt.imshow(image)
#     plt.show()

valid_generator = datagen.flow_from_dataframe(
                                        dataframe = valid_set,
                                        directory="",x_col="Path",
                                        y_col=types,
                                        class_mode="raw",
                                        color_mode="rgb",
                                        target_size=(224,224),
                                        batch_size=8)

inputShape = (224, 224, 3)
model1 = tf.keras.models.Sequential([
    tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_shape=inputShape),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(14, activation='sigmoid')
])


chanDim = -1

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=inputShape),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(3, 3),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(axis=chanDim),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation='sigmoid')
])


model1.compile(optimizer=Adam(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


date = datetime.now().strftime("_%m_%d_%Y_%H_%M_%S")

csv_logger = CSVLogger('logs/log_lateral'+ date +'.csv')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.1, patience=2, mode='min')
model_path = 'saved_models/best_model_lateral'+date+'.h5'
mc = ModelCheckpoint(model_path, monitor='val_loss',  mode='min', verbose=1)
history = model1.fit_generator(train_generator, epochs = 15,
                               steps_per_epoch = 136,
                               validation_data=valid_generator,
                               validation_steps=4,
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