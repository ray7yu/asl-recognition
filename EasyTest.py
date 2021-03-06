# -*- coding: utf-8 -*-
"""Hard.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PLSIpmEo0L-AJGfv98SXpPXMvbtCTqtT
"""

def EasyTest (X):
  from tensorflow.keras.applications.vgg16 import preprocess_input
  from tensorflow.keras.models import load_model
  from skimage.transform import resize
  from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
  from sklearn.preprocessing import OneHotEncoder
  import numpy as np


  model = load_model('./model1.h5')
  print("model 1 loaded")

  X = preprocess_input(X)
  X_resized = np.array([resize(image, (96, 96)) for image in X])

  y_pred = model.predict(X_resized)

  # Convert output to numbers
  y_pred = np.argmax(y_pred,axis=1)+1

  return y_pred