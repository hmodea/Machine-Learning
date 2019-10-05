import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from keras.layers import Conv2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from generator import generate


input_shape = (64,64,3)
#Hyperparameters
batch_size = 128
epochs = 10
wait_epochs = 3
drop_prob = 0.4

def nvidia_model(drop_prob):
    
    model = Sequential()
    # normalization
    model.add(Lambda(lambda x: x/127.5 - 1, input_shape=input_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(drop_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse", metrics=['mse'])
    
    return model

if __name__ == '__main__':
    
    np.random.seed(10)
    
    # import data for training
    data = pd.read_csv("./modified_driving_log.csv")
    
    training_data, validation_data = train_test_split(data, test_size=0.3)
    
    train_generator = generate(training_data, batch_size=batch_size)
    validation_generator = generate(validation_data, batch_size=batch_size)
    
    # callbacks
    checkpoint = ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0.0003, patience=wait_epochs)
    
    model = nvidia_model(drop_prob)
    
    model.fit_generator(train_generator, \
                steps_per_epoch=np.ceil(len(training_data)/batch_size), \
                validation_data=validation_generator, \
                validation_steps=np.ceil(len(validation_data)/batch_size), \
                callbacks = [checkpoint, stopper], \
                epochs=epochs, verbose=1)
    
    model.save('model.h5')
    
    
    
    
    
    

