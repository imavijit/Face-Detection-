#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import keras
from matplotlib import pyplot as plt
from PIL import Image
from keras.datasets import cifar10
from keras.utils import np_utils


# In[2]:


# load the data
(X_train, y_train),(X_test, y_test) = cifar10.load_data()


# In[3]:


#dataset characteristics
print("Training Images : {}".format(X_train.shape))
print("Test Images : {}".format(X_test.shape))

# Single image prooperties
print(X_train[0].shape)


# In[32]:


# plot some images
for i in range(0,9):
    plt.subplot(330 + 1 + i)
    img = X_train[500 + i]
    plt.imshow(img)

plt.show()


# In[5]:


#preprocessing the dataset 

#fixed random seed for same random output
seed=6
np.random.seed(seed)

#load the data 
(X_train, y_train), (X_test,y_test) = cifar10.load_data()

#normalize the inputs from 0 -255 to  0.0 - 1.0
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0


# In[6]:


# class label shape
print(y_train.shape)
print(y_train[0])


# In[7]:


# one hot encoding ; [4] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] or [6] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

#encoding
Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print(Y_train.shape)
print(Y_train[0])

#num of classes
num_classes=Y_test.shape[1]
print("no. of classes : ", num_classes)


# In[8]:


# import necessary things from keras
from keras.models import Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, Dropout, Activation
from keras.optimizers import SGD


# Table 1: The All-CNN network used for classification on CIFAR-10. Original Paper:
# https://arxiv.org/pdf/1412.6806.pdf  - Striving for simplicity<br>
#     
# Model- C
# * Input 32 × 32 RGB image
# * 3 × 3 conv. 96 ReLU
# * 3 × 3 conv. 96 ReLU
# * 3 × 3 max-pooling stride 2
# * 3 × 3 conv. 192 ReLU
# * 3 × 3 conv. 192 ReLU
# * 3 × 3 max-pooling stride 2
# * 3 × 3 conv. 192 ReLU
# * 1 × 1 conv. 192 ReLU
# * 1 × 1 conv. 10 ReLU
# * global averaging over 6 × 6 spatial dimensions
# * 10 (or 100)-way softmax

# In[9]:


#define the model 
def all_cnn(weights = None):
    
    model = Sequential()
    
    #add layers
    model.add(Conv2D(96, (3, 3),input_shape = (32,32,3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3),padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1),padding = 'valid'))
    
    #add global average pooling layer with softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    #load the weights
    if weights:
        model.load_weights(weights)
        
    #return the model
    return model 


# In[10]:


#defining the hyperparameters
learning_rate = 0.01
weight_decay  = 1e-6
momentum = 0.9

#build model
model = all_cnn()

#defining optimizers and compile model
sgd = SGD(lr = learning_rate, decay = weight_decay, momentum= momentum, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer=sgd, metrics= ['accuracy'])

print(model.summary())

#Additional Parameters+
epochs= 350
batch_size = 32

#fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = epochs, batch_size = batch_size, verbose =1)


# In[15]:


#check accuracy of model
scores = model.evaluate(X_test, Y_test,batch_size=32)
print("Accuracy: {}".format(scores[1]))


# In[16]:


model.save_weights("all_cnn_cifar10.hd5")


# In[18]:


#dictionary of class labels and names
classes = range(0,10)

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class_labels = dict(zip(classes, labels))
print(class_labels)


# In[33]:


#generate batch of 9 to prdict
batch = X_test[500:509]
#actual labels
labels = np.argmax(Y_test[500:509], axis= -1)

#make predictions
prediction = model.predict(batch, verbose = 1)

print(prediction)


# In[37]:


#use np.argmax() to convert class probabilities to class labels
class_result = np.argmax(prediction, axis = -1)
#predicted labels
print(class_result)

#actual labels
print(labels)


# In[35]:


# create a grid of 3 x 3 images
fig, axes=plt.subplots(3, 3, figsize=(20,6))
fig.subplots_adjust(hspace = 1)
axes= axes.flatten()

for i, img in enumerate(batch):
     for key, value in class_labels.items():
        if(class_result[i] == key):
            title='Prediction: {} \nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axes[i].set_title(title)
            axes[i].axes.get_xaxis().set_visible(False)
            axes[i].axes.get_yaxis().set_visible(False)
            
     axes[i].imshow(img)
    
plt.show()
            
           


# In[ ]:




