# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Readme_images/initial_distribution.png "Initial steering angle distribution"
[image2]: ./Readme_images/adapted_distribution.png "Adapted steering angle distribution"
[image3]: ./Readme_images/Sample_image.png "Sample Image"
[image4]: ./Readme_images/flipped_image.png.png "Flipped Image"
[image5]: ./Readme_images/noised_image.png "Noised Image"
[image6]: ./Readme_images/Inverted_channels_image.png "Inverted Channels Image"
[image7]: ./Readme_images/altered_brightness_image.png "Altered Brightness Image"
[image8]: ./Readme_images/nvidia_model_architecture.png "Nvidia model architecture"
  
---

#### Project Structure

My project includes the following files:
* preprocessing.py includes data exploration and offline preprocessing including augmentation.
* generator.py includes the script to generate training and validation batches using a python generator along with online preprocessing.
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network based on nvida end to end learning for self-driving cars model architecture.
* writeup_report.md summarizing the results


### Model Architecture and Training Strategy

#### 1. Data collection strategy

The data provided by Udacity was used as basis to train the model in this project.

#### 2. Data exploration and preprocessing

When exploring the data provided it was observed that it follows a gaussian distribution with a somehow narrow standard deviation. The figure below depicts the initial distribution:

![alt text][image1]

It is clear that images corresponding to steering angles greater than absolute 0.25 are under represented in the data set and thus it is likely that the trained model would have a suboptimal performance for similar cases.
First, in order to increase the size of our data set I made use of the the left and right camera images as well as the center one. An experimental correction factor of ±0.25 was applied to the steering angles of those cameras.
Second, in order to increase the representation of large steering angles I took the samples having angles greater than absolute 0.25 and performed four augmentation operation independently on them namely flipping, addition of gaussion noise, altering image brightness and color channel inversion. An example for each augmentation operation performed on a sample image is provided in the figures below:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

We finally ended up with a distribution that is more close to a uniform distribution than it was before.

![alt text][image2]


#### 3. Model architecture

The adopted model to tackle this problem was based on the nvidia end to end learning for self driving car [Link](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The reason is that the architecture was previously built to tackle a similar problem thus it was a good candidate for a solution to the problem beforehand. The model architecture is showen below:

![alt text][image8]

The adopted architecture had however, a slight difference to the one above as the 1164 neurons fully-connected layer has been removed to decrease the number of model parameters thus, making it more convenient to train the model with the size of dataset available. Furthermore, a dropout layer is added following the convolutional layers to cater against overfitting.

#### 4. Training Strategy

The following hyperparameters were chosen after multiple trials to train the model:
*batch size = 128
*epochs = 10
*patience for early stopping = 3
*dropout probability = 0.4
* optimizer : Adam optimizer.

Before the image is input to the model it is subjected to an online preprocessing pipeline which includes: 
*Cropping layer : to crop the hood and the horizon.
* Resizing layer: to resize the image to the expected size by the model (64,64,3).
* Color channel conversion: The image is converted to the YUV color space which is expected by the nvidia model.
* Normalization: Normalize the pixels to have zero mean and unit standard deviation


#### 5. Output video

The model is then utilized to drive the car autonomously in the simulator using the drive.py script. It is worth noting that the images from the simulator and subjected to the same preprocessing pipeline before the steering angle is predicted.

Finally, a video for the self-driving car on the first track is generated [video](./output_video.mp4)using the video.py script.


#### 6. Possible further improvements

* Use an RNN to the output of the CNN: one drawback of the solution provided is that the model lacks the temporal component i.e the steering angle of the car at time t-1 has no bearing on the angle at time t one of the solution to that problem is to feed the output steering angles from the CNN for a chosen window size to a LSTM or GRU.

* Track 2: Collect data for the more challenging track 2 and train the model and both track 1 and track 2 data to help it to generalize better to different roads and driving situations.

* Transfer learning: Make use of imagenet trained models to tackle the problem beforehand. A good candidate lightweight model to try could be MobileNet.

