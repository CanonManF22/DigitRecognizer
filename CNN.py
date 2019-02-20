import numpy as np # linear algebra
import pandas as pd 
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from sklearn.model_selection import train_test_split

#dimensions of image
image_width = image_height = 28

#use Sequential model from TF
model = Sequential()
model.add(Convolution2D(28, 1, input_shape = (image_width, image_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(28, 28, 1, input_shape = (image_width, image_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(28, 28, 1, input_shape = (image_width, image_height, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

#experiment with dropout
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

training_data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train, test = train_test_split(training_data, test_size = .33, random_state = 42)

labels = training_data['label']

train_final = np.expand_dims(train, axis=-1)
test_final = np.expand_dims(test, axis=-1)

print(train_final.shape, test_final.shape)
num_images = len(train_final)
train_final.reshape(num_images, 785, 1)

model.fit(train_final, train['label'], epochs=10, batch_size=100)

#save weights for restart
model.save_weights('CNN_weights.h5')

test_final.reshape(num_images, 785, 1)
score = model.evaluate(test_final, test['label'])
print(score)