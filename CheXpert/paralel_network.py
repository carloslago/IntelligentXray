import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd


train_dir = os.path.join('CheXpert-v1.0-small/train')
val_dir = os.path.join('CheXpert-v1.0-small/valid')
training_prev = pd.read_csv("CheXpert-v1.0-small/train.csv")
training_imgs = list(training_prev['Path'])
types = ['No_Finding', 'Enlarged_Cardiomediastinum', 'Cardiomegaly', 'Lung_Opacity', 'Lung_Lesion', 'Edema',
         'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural_Effusion', 'Pleural Other',
         'Fracture', 'Support Devices']


training_set = pd.DataFrame({'Images': training_imgs})
for t in types[:5]:
    training_set[t] = list(training_prev[t.replace('_', ' ')])
    training_set[t] = training_set[t].astype(int)
    training_set[t] = training_set[t].astype(str)



training_set['New_class'] = training_set['No_Finding'] + training_set['Enlarged_Cardiomediastinum']  + \
                            training_set['Cardiomegaly'] + training_set['Lung_Opacity'] + \
                            training_set['Lung_Lesion']

train_dataGen = ImageDataGenerator(rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,)

train_generator = train_dataGen.flow_from_dataframe(
                                        dataframe = training_set,
                                        directory="",x_col="Images",
                                        y_col="New_class",
                                        class_mode="categorical",
                                        target_size=(224,224),
                                        batch_size=8)

print(train_generator.class_indices)


# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(filters = 56,kernel_size = (3,3), activation = 'relu', input_shape = (224,224,3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(8, activation='softmax')
# ])
#
# model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])
#
# model.fit_generator(train_generator, epochs = 10, steps_per_epoch = 20 )

# datagen = ImageDataGenerator(rescale=1/255)
# training_datagen = ImageDataGenerator(rescale=1/255, rotation_range=30, zoom_range=0.3)

# train_generator = datagen.flow_from_directory(
#     directory=train_dir,
#     target_size=(224, 224),
#     color_mode="grayscale",
#     batch_size=64,
#     class_mode="categorical"
# )
#
# validation_generator = datagen.flow_from_directory(
#     directory=val_dir,
#     target_size=(224, 224),
#     color_mode="grayscale",
#     batch_size=64,
#     class_mode="categorical"
# )

'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(axis=1),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.0001, decay=1e-5),
              metrics=['acc'])

# Fitting the model, in this case using the test_generator to plot afterwards the results.
history = model.fit_generator(generator=train_generator,
                    validation_data=test_generator,
                    steps_per_epoch=67,
                    validation_steps=22,
                    epochs= 15,
                    verbose = 1
)

model.save('t_network.h5')


# Plot training and test acc and loss
print(history.history.keys())
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
'''