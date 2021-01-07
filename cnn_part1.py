
#Importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialize the CNN
classifier = Sequential()

#Create Conv & Relu layer
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'))

classifier.add(Convolution2D(32,3,3, activation='relu'))

#Create Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Flattening Layer
classifier.add(Flatten())

#Fully Connected Layer
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))


#Compiling the CNN 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


#Fitting the CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_dataset = ImageDataGenerator(rescale=1./255)

test_dataset = ImageDataGenerator(rescale=1./255)


train = train_dataset.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size=32, class_mode='binary')

test = test_dataset.flow_from_directory('dataset/test_set', target_size = (64,64), batch_size=32, class_mode='binary')

#classifier.fit_generator(train, samples_per_epoch = 7999, nb_epoch = 500, validation_data = test, nb_val_samples=2000)


classifier.fit_generator(train, samples_per_epoch = 8000, nb_epoch = 500, validation_data = test, nb_val_samples=2000)




##############################################################
##############################################################


import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog.jpg', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

print(result)
result
train.class_indices

if result[0][0]==1:
    prediction='dogs'
else:
    prediction='cats'
    
    
print(prediction)






















