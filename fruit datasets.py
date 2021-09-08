import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import tensorflow as tf
from tensorflow import keras
from tf.keras.preprocessing.image import ImageDataGenerator
train_data_dir = r'D:\Dataset\fruits\fruits-360\Training'
validation_data_dir = r'D:\Dataset\fruits\fruits-360\Test'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(32,32),
        batch_size =16,
        class_mode='categorical',
        shuffle=True)
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(32,32),
        batch_size=16,
        class_mode='categorical',
        shuffle=False)

model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3), padding ='same', input_shape=(32,32,3), activation=('relu')),
    keras.layers.Conv2D(32,(3,3), activation ='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64,(3,3), activation =('relu')),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(131, activation='softmax'),    
    ])
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'Adam',
              metrics = ['accuracy'])
model.fit( train_generator, epochs =10)