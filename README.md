# Final Project
Two models are included in the final project. Model 1 should be used for the easy test and model 2 should be used for the hard test. ```vgg_16_transfer_learning_model_1.ipynb``` contains model 1 training. ```vgg_16_transfer_learning_model_2.ipynb``` contains model 2 training. 


### Required Packages
Tensorflow 1.15 is needed in order to load models successfully. If tensorflow 1.14 is used, you will encounter the below error message
```
ValueError: ('Unrecognized keyword arguments:', dict_keys(['ragged']))
```
Upgrading Tensorflow to the right version will solve this issue.

### How to Train the Models
```train.py``` contains the function ```trainModel(X, Y)``` which takes an input data set X and a desired output vector Y as inputs and returns the trained model. Below code snippet shows how to import and use this function.
```
from train import trainModel
model = trainModel(data, labels)
```
Even though we have 2 slightly different models, both models could use the same function above for training. The number of neurons in the output layer depends on the number of classes in the training data. Thus, it is critical to include all but not excessive number of classes (unknown class is an exception) in the training set. For example, since the easy test set has 4 classes, the training data for that model should contain the same 4 classes. For the hard test set, all 9 classes (A to I) should be included in the training set. Unknown images should **not** be included in the training set as the unknown class is determined by a certain threshold on the output layer. 

All the hyper parameter values are listed at the top of the file and can be modified:
```
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
```

### How to Test Models
The models are saved in HDF5 format. They unfortunately could not be uploaded to this repo because the file size is too large. They are stored in Google Drive and could be downloaded following the below instructions.
There are 2 test functions, one for each model. One should be used for the easy test set and the other should be used for the hard test set. The EasyTest() can be loaded from EasyTest.py and HardTest() can be loaded from HardTest.py.

#### Model 1
To test model 1, download ```model1.h5``` from [model 1 link](https://drive.google.com/open?id=1eneYXqnWpvsBu_X5epXYZrE1L8mCz2ob) and save it in the same directory as files from this repo.
The function ```EasyTest(X)``` where X is the input data set should be used for the easy test set. This function uses model 1 to predict the labels and outputs a set of labels.
```
from EasyTest import EasyTest
y_pred = EasyTest(X)
```

#### Model 2
To test model 2, download ```model2.h5``` from [model 2 link](https://drive.google.com/open?id=186_lJU5BJBJw9naMto_pfa5E4AdwAv8Q) and save it in the same directory as files from this repo.
The function ```HardTest(X)``` where X is the input data set should be used for the hard test set. This function uses model 2 to predict the labels and outputs a set of labels.
```
from HardTest import HardTest
y_pred = HardTest(X)
```

### How to Test Models [deprecated]
The models are saved in HDF5 format. They unfortunately could not be uploaded to this repo because the file size is too large. They are stored in Google Drive and could be downloaded following the below instructions.

#### Model 1
To test model 1, download ```model1.h5``` from [model 1 link](https://drive.google.com/open?id=1eneYXqnWpvsBu_X5epXYZrE1L8mCz2ob) and save it in the same directory as files from this repo. Then load the model by running code in ```model_1_test.ipynb```. The data directory in the 3rd cell should be changed to your test data directory in order to load the correct data. Nothing else needs to be changed and the last code cell should output the labels of the test set. This model should only be used on the easy test set since it only outputs 4 classes.ok


#### Model 2
To test model 1, download ```model2.h5``` from [model 2 link](https://drive.google.com/open?id=186_lJU5BJBJw9naMto_pfa5E4AdwAv8Q) and save it in the same directory as files from this repo. Then load the model by running code in ```model_2_test.ipynb```. The data directory in the 3rd cell should be changed to your test data directory in order to load the correct data. Nothing else needs to be changed and the last code cell should output the labels of the test set.
