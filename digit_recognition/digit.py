from operator import imod
from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.core.algorithms import mode
import sklearn
import sklearn.model_selection
from sklearn import preprocessing
from keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense,MaxPooling2D,Convolution2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.ops.gen_math_ops import mod
import random

# will list all files under the input directory
dataDir='C:/Users/Besitzer/Desktop/digit_recognizer'
for i in os.walk(dataDir):
    print(i)

#reading the data
train=pd.read_csv('C:/Users/Besitzer/Desktop/digit_recognizer/train.csv')
test=pd.read_csv('C:/Users/Besitzer/Desktop/digit_recognizer/test.csv')
# print(train.head())
# print(train.describe())


##Data processing
X_train=train.drop(labels=['label'],axis=1)
y_train=train['label']
# print(X_train.shape)
# print(y_train.shape)

#normalization

X_train=X_train/255
X_test=test/255

#convert lable into OneHotEncoder
#y_train=to_categorical(y_train ) 
#or 
y_train=pd.get_dummies(y_train)
# print(y_train.shape)

#reshaping of images
X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)
# print(X_train.shape)
# print(X_train.shape)


#spliting data into trainging and validation set

X_train,X_valid,y_train,y_valid=sklearn.model_selection.train_test_split(X_train,y_train,test_size=0.2,random_state=50)

# print(X_train.shape)
# print(y_train.shape)
# print(X_valid.shape)
# print(y_valid.shape)

##Model
def LeNet_model():
    model=Sequential()
    model.add(Convolution2D(64,kernel_size=(5,5),input_shape=(28,28,1),activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10,activation='softmax'))
    model.compile(Adam(0.01),loss='categorical_crossentropy',metrics=['accuracy'])   
    return model
    

model=LeNet_model()
print(model.summary())

h=model.fit(x=X_train,y=y_train,batch_size=500,epochs=20,verbose=1,validation_data=(X_valid,y_valid),shuffle=True)

plt.figure(figsize=(5,6))
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.axis('off')
plt.legend(['accuracy','val_accuracy'])
plt.title('Accuracy',loc='center',fontSize=10)
# plt.show()

plt.figure(2,figsize=(5,6))
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.axis('off')
plt.legend(['loss','val_loss'])
plt.title('Losses',loc='center',fontSize=10)
# plt.show()

# score=model.evaluate(X_test,y_test,verbose=0)
# print(type(score))
# print('Test score:',score[0])
# print('Test accuracy:',score[1])

prediction=np.argmax(model.predict(X_test),axis=1)
prediction=pd.Series(prediction,name='label')
submission=pd.concat([pd.Series(range(1,28001),name='ImageID'),prediction],axis=1)
submission.to_csv('C:/Users/Besitzer/Desktop/digit_recognizer/tested_result.csv',index=False)