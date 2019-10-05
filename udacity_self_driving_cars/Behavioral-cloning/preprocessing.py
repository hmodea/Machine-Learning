#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[12]:


logs = pd.read_csv("driving_log.csv")


# In[14]:


logs['rounded_steering']= np.around(logs['steering'],2)


# In[15]:


max_steering_angle = logs['steering'].max()
min_steering_angle = logs['steering'].min()

print('Maximum steering angle = {:.3f}'.format(max_steering_angle))
print('Minimum steering angle = {:.3f}'.format(min_steering_angle))


# In[19]:



def convert_to_three_camera_mode(logs, angle_correction = 0.25):
    
    columns =['image','steering_angle']
    df_steering = pd.DataFrame(np.zeros((len(logs)*3, 2)),columns=columns)
    
    for i in range(0,len(logs)):
        
        df_steering.loc[3*i,'image'] = logs.loc[i,'center']
        df_steering.loc[3*i+1,'image'] = logs.loc[i,'left'] 
        df_steering.loc[3*i+2,'image'] = logs.loc[i,'right']
        
        df_steering.loc[3*i,'steering_angle'] = logs.loc[i,'steering']
        df_steering.loc[3*i+1,'steering_angle'] = logs.loc[i,'steering'] + angle_correction
        df_steering.loc[3*i+2,'steering_angle'] = logs.loc[i,'steering'] - angle_correction
        
    return df_steering
    


# In[21]:


df_3_cameras = convert_to_three_camera_mode(logs)


# In[12]:


df_3_cameras['steering_angle_rounded'] = np.around(df_3_cameras['steering_angle'],2)


# ### Augmentation

# Offline augmentation is performed in this step with the aim to balance the data step, since large steering angles are observed to be minimal in this dataset

# In[8]:


from imgaug import augmenters as iaa
import cv2


# In[38]:


def flip_images(df_input,df_original):
    
    flip = iaa.Fliplr(1.0)
    
    for i in range(len(df_input)):
        
        index = df_input.index[i]
        
        image = str(df_input.loc[index,'image']).strip()
        img = cv2.imread(image)
    
        flipped_img = flip.augment_image(img)
        updated_angle = - df_input.loc[index,'steering_angle']
        rounded_angle = np.around(updated_angle,2)
        
        
        saved_name = str(df_input.loc[index,'image']).strip('jpg').strip('.') + "_" + "flipped" + "." + "jpg"
        df_original = df_original.append(pd.DataFrame({"image":[saved_name],"steering_angle":[updated_angle],"steering_angle_rounded":[rounded_angle]}), ignore_index = True)
        cv2.imwrite(saved_name,flipped_img)
        
    return df_original
        
    


# In[39]:


df_large_steering_angles = df_3_cameras[np.absolute(df_3_cameras['steering_angle_rounded']) > 0.25]


# In[41]:


df_with_flipped_images = flip_images(df_large_steering_angles,df_3_cameras)


# In[45]:


def noise_images(df_input,df_original):
    
    additive_gaussian_noise = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
    
    for i in range(len(df_input)):
        
        index = df_input.index[i]
        
        image = str(df_input.loc[index,'image']).strip()
        img = cv2.imread(image)
        
        noisy_img = additive_gaussian_noise.augment_image(img)
        angle = df_input.loc[index,'steering_angle']
        rounded_angle = df_input.loc[index,'steering_angle_rounded']
        
        
        saved_name = str(df_input.loc[index,'image']).strip('jpg').strip('.') + "_" + "noised" + "." + "jpg"
        df_original = df_original.append(pd.DataFrame({"image":[saved_name],"steering_angle":[angle],"steering_angle_rounded":[rounded_angle]}), ignore_index = True)
        cv2.imwrite(saved_name,noisy_img)
        
    return df_original
        


# In[46]:


df_with_noise = noise_images(df_large_steering_angles,df_with_flipped_images)


# In[49]:


def invert_pixels(df_input,df_original):
    
    invert = iaa.Invert(1.0)
    
    for i in range(len(df_input)):
        
        index = df_input.index[i]
        
        image = str(df_input.loc[index,'image']).strip()
        img = cv2.imread(image)
        
        inverted_img = invert.augment_image(img)
        angle = df_input.loc[index,'steering_angle']
        rounded_angle = df_input.loc[index,'steering_angle_rounded']
        
        
        saved_name = str(df_input.loc[index,'image']).strip('jpg').strip('.') + "_" + "inverted" + "." + "jpg"
        df_original = df_original.append(pd.DataFrame({"image":[saved_name],"steering_angle":[angle],"steering_angle_rounded":[rounded_angle]}), ignore_index = True)
        cv2.imwrite(saved_name,inverted_img)
        
    return df_original


# In[50]:


df_with_inversion = invert_pixels(df_large_steering_angles,df_with_noise)


# In[52]:


def alter_brightness(df_input,df_original):
    
    bright_dark = iaa.Multiply((0.5, 1.5), per_channel=0.5)
    
    for i in range(len(df_input)):
        
        index = df_input.index[i]
        
        image = str(df_input.loc[index,'image']).strip()
        img = cv2.imread(image)
        
        altered_img = bright_dark.augment_image(img)
        angle = df_input.loc[index,'steering_angle']
        rounded_angle = df_input.loc[index,'steering_angle_rounded']
        
        
        saved_name = str(df_input.loc[index,'image']).strip('jpg').strip('.') + "_" + "bright" + "." + "jpg"
        df_original = df_original.append(pd.DataFrame({"image":[saved_name],"steering_angle":[angle],"steering_angle_rounded":[rounded_angle]}), ignore_index = True)
        cv2.imwrite(saved_name,altered_img)
        
    return df_original


# In[53]:


df_with_altered_brightness = alter_brightness(df_large_steering_angles,df_with_inversion)


# In[56]:


df_with_altered_brightness.to_csv(path_or_buf="./modified_driving_log.csv")


# In[58]:


read_file = pd.read_csv("./modified_driving_log.csv")


# In[ ]:


print(len(df_with_altered_brightness))

