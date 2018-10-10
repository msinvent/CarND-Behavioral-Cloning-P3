# coding: utf-8
'''
@Programmer : Manish Sharma
@description : code to train an end to end deep neural network to drive an autonomous vehicle on unity simulator provided by udacity
Main Section of the codes are :
    1 - DataGenerator class : inherits from keral.utils.Sequence to allow picking data on the fly to be fed to DNN to overcome ram limitations
    2 - Deep Neural Network description in keras
    3 - Deep Neural Network specicifications and training to finally save the model
'''

# In[1]:


import cv2, numpy as np
import pandas as pd


# In[2]:


# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# https://github.com/shervinea/enzynet
import keras

'''
@DataGenerator class written to modify the keral standard DataGenerator, I found this approach extremely streamlined and clean.
Class randomize the input sample and reads only specified number of image samples in memory to feed. This class supports multiprocessing
but requires careful calculation because you may wind up overloading your RAM if your batch size and number of processors is large.
Thanks to Safeer Afaque for this recommendation over the use of yield
'''
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), shuffle=False, objectName = 'defaultName', **params):
#         'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.indexes = list_IDs.index
        self.relativeLocation  = relativeLocation
        self.objectName = objectName

    def __len__(self):
#         'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
#         'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         print(self.objectName, '\n\n', indexes)
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def __on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         print(self.objectName, ': original indexes order : ',self.indexes)
#         print(self.objectName, '\n')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
#         print(self.objectName, ': reshuffling happening here, new index order is : \n\n',self.indexes)
            print('shuffling happening here')
        
    def __data_generation(self, indexes):
#         'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size))
        
        i = 0
        for index in indexes:
            try:
                imageName = self.list_IDs[index]
                # index missing thus throw indexError
                steeringInput = self.labels[index]
                
                if(pd.isnull(imageName) or pd.isnull(steeringInput)):
#                     print(imageName,steeringInput)
                    raise IndexError()
                
                try:
                    image_path = self.relativeLocation + 'IMG/' + imageName.split('/')[-1]
                    image = cv2.imread(image_path).reshape((1,160,320,3))
               
                    if image is None:
                        raise IndexError()
                                    
                    X[i,] = image
                    y[i,] = steeringInput
                    i = i+1
                except IndexError:
                    print(self.objectName,' ,inner index ',index,' Either image or steering angle is missing')
                    pass
                except KeyError:
                    print(self.objectName,' ,inner key ',index,' Either image or steering angle is missing')
                    pass
        
        
        
            except IndexError:
                print(self.objectName,' ,outer index ',index,' Either image or steering angle is missing')
                pass
            except KeyError:
                print(self.objectName,' ,outer key ',index,' Either image or steering angle is missing')
                pass
        
        return X, y


# In[3]:
'''
@Declarions of DataGenerator objects for training_generator and validation_generator, sklearn train_test_split is used to separate training and
validation dataset
'''

from sklearn.model_selection import train_test_split

# Parameters
params = {'dim': (160,320,3),
          'batch_size': 32,
          'shuffle': True,
          'relativeLocation':'../assignment_3/data_downloaded/data/'}

# Datasets
relativeLocation = params['relativeLocation']
df = pd.read_csv(relativeLocation + 'driving_log.csv')
X = df['center']
y = df['steering']

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

training_generator = DataGenerator(X_train, y_train, objectName = 'trainingGenerator', **params)
validation_generator = DataGenerator(X_validation, y_validation, objectName = 'validationGenerator', **params)

# Some Debugging happening here
XX, yy = training_generator.__getitem__(0)
XX, yy = validation_generator.__getitem__(0)
# print(yy)
# for image in XX:
#     plt.figure(figsize=(10,10))
#     plt.imshow(image) # This is showig the image thinking it is a bgr image
#     print(image)
    


# In[4]:
'''
Keral library imports
'''
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Dropout


# In[5]:
'''
Keral model description
'''

# ch, row, col = 3, 80, 320  # Trimmed image format
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) #Normalization layer
model.add(Cropping2D(cropping=((60,15),(0,0))))

model.add(Conv2D(24, activation="relu", kernel_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(36, activation="relu", kernel_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(48, activation="relu", kernel_size=(3, 3), strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))

model.add(Dense(30))
model.add(Dropout(0.25))

model.add(Dense(1))

# printing the model description to take a snapshot and print it to the project
model.summary()

# # Network optimization choices set to mse and adam optimizer
model.compile(loss='mse',optimizer='adam')


## In[6]:
#
#
## Train model on dataset
#
#model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=False,
#                    shuffle = True,
##                     steps_per_epoch = 100,
#                    epochs = 8,
#                    workers=4,
#                    verbose=2)
#
#model.save('model_generators_8E.h5')
#print('model trained with 8 epochs')
#
## In[7]:
#
#
#model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=False,
#                    shuffle = True,
##                     steps_per_epoch = 100,
#                    epochs = 8,
#                    workers=4,
#                    verbose=2)
#
#model.save('model_generators_16E.h5')
#model.save('model.h5')
#print('model trained with 16 epochs')
#
#
## In[8]:
#
#
#model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=False,
#                    shuffle = True,
##                     steps_per_epoch = 100,
#                    epochs = 8,
#                    workers=4,
#                    verbose=2)
#
#model.save('model_generators_24E.h5')
#print('model trained with 24 epochs')
#
## In[9]:
#
#
#model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=False,
#                    shuffle = True,
##                     steps_per_epoch = 100,
#                    epochs = 8,
#                    workers=4,
#                    verbose=2)
#
#model.save('model_generators_32E.h5')
#print('model trained with 32 epochs')
#
## In[10]:
#
#
#model.fit_generator(generator=training_generator,
#                    validation_data=validation_generator,
#                    use_multiprocessing=False,
#                    shuffle = True,
##                     steps_per_epoch = 100,
#                    epochs = 8,
#                    workers=1,
#                    verbose=2)
#
#model.save('model_generators_40E.h5')
#print('model trained with 40 epochs')
