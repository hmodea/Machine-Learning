import pandas as pd
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import img_to_array, load_img
#from keras.applications.densenet import preprocess_input

def convert_to_YUV(img):
    
    return cv2.cvtColor(img,cv2.COLOR_RGB2YUV)

def crop(img):
    
    return img[55:135, :, :]

def resize(img, w_image = 64, h_image = 64):
    
    return cv2.resize(img, (w_image, h_image))

def preprocess(img):
    
    img = convert_to_YUV(img)
    img = crop(img)
    img = resize(img)
    
    return img

def generate(examples, batch_size):
    
    num_examples = len(examples)

    while 1:
        
        shuffle(examples)
        
        for offset in range(0, num_examples, batch_size):
            
            batch = examples[offset:offset + batch_size]
            
            images = []
            for _ , sample in batch.loc[:].iterrows():
                
                path = load_img(sample['image'].strip())
                img = img_to_array(path)
                img = preprocess(img)
                images.append(img)
  
            X = np.array(images)
            y = np.array(batch['steering_angle'])
            
            yield shuffle(X,y)
            