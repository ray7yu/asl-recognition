from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
import sys

# train validation split ratio
split_ratio = 0.2
# batch size
batch_size = 150
# steps per iteration
steps_per_epoch = 40
# transfer learning learning rate
transfer_learning_lr = 1e-3
# transfer learning number of iterations
transfer_learning_epochs = 300
# fine tuning learning rate
fine_tuning_lr = 1e-7
# fine tuning number of iterations
fine_tuning_epochs = 500


def preprocess(X, y):
  # Pre-process and resize images
  X = preprocess_input(X)
  X = np.array([resize(image, (96, 96)) for image in X])

  # Use one-hot encoding for labels
  onehot_encoder = OneHotEncoder(sparse=False)
  one_hot_y = onehot_encoder.fit_transform(y) # may need to reshape with reshape(-1, 1)

  return X, one_hot_y

def trainModel(data, label):
  X, one_hot_y = preprocess(data, label)

  # Train validation split
  X_train, X_val, y_train, y_val = train_test_split(X, one_hot_y, test_size=split_ratio)

  # Training data generator
  datagen_train = ImageDataGenerator(
        zoom_range=[0.8,1.0], 
        horizontal_flip=True, 
        vertical_flip=True, 
        rotation_range=90
        )

  # Validation data generator
  datagen_val = ImageDataGenerator()

  generator_train = datagen_train.flow(x=X_train, y=y_train,
                                      batch_size=batch_size,
                                      shuffle=True)

  generator_val = datagen_val.flow(x=X_val, y=y_val,
                                    batch_size=batch_size,
                                    shuffle=False)

  steps_val = generator_val.n / batch_size

  IMG_SIZE = 96
  IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

  # Load VGG-16 model
  base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
  base_model.trainable = False

  num_classes = one_hot_y.shape[1]

  model = Sequential ([
      base_model,
      # Flatten the output of the VGG16 model because it is from a
      # convolutional layer.
      Flatten(),
      # Add 2 fully-connected layers.
      # This is for combining features that the VGG16 model has
      Dense(4608, activation='relu'),
      Dense(1024, activation='relu'),   
      # Add a dropout-layer which may prevent overfitting and
      # improve generalization ability to unseen data
      Dropout(0.5),
      # Add the final layer for the actual classification.
      Dense(num_classes, activation='softmax')                          
  ])

  model.compile(optimizer=Adam(lr=transfer_learning_lr), loss='categorical_crossentropy', metrics=['accuracy'])

  early_stopping = EarlyStopping(monitor='val_loss', mode='min', 
                                patience=30, restore_best_weights=True, 
                                verbose=1)

  history = model.fit(x=generator_train, epochs=transfer_learning_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=generator_val,
                      validation_steps=steps_val,
                      callbacks=[early_stopping])

  # Unfreeze the pre-trained model for fine-tuning
  base_model.trainable = True

  for layer in base_model.layers:
      # Boolean whether this layer is trainable.
      trainable = ('block5' in layer.name or 'block4' in layer.name)
      
      # Set the layer's bool.
      layer.trainable = trainable

  model.compile(optimizer=Adam(lr=fine_tuning_lr), loss='categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(x=generator_train, epochs=fine_tuning_epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_data=generator_val,
                      validation_steps=steps_val,
                      callbacks=[early_stopping])

  return model